#!/bin/bash
# Captures a 10-second rosbag from a running sensor suite container on a remote server
# and copies it to the current directory.
#
# Usage:
#   ./capture_bag.sh user@server --serial vss_016
#   ./capture_bag.sh dockware@beyonce --serial vss_016
#   ./capture_bag.sh dockware@beyonce --serial vss_016 --duration 30
#
# Options:
#   --serial <serial>      Sensor suite serial (e.g. vss_016). vss_ prefix auto-added if missing.
#   --duration <seconds>   Recording duration in seconds (default: 10)

set -e

if [ -z "$1" ]; then
    echo "Error: Missing target (user@server)"
    echo "Usage: $0 user@server --serial <serial> [--duration <seconds>]"
    exit 1
fi

TARGET="$1"
shift

SERIAL=""
DURATION=10

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --serial)
            SERIAL="$2"
            shift
            ;;
        --duration)
            DURATION="$2"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

if [ -z "$SERIAL" ]; then
    echo "Error: --serial is required."
    exit 1
fi

if [[ ! "${SERIAL}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Security Error: SERIAL '${SERIAL}' contains invalid characters."
    exit 1
fi

if [[ ! "${DURATION}" =~ ^[0-9]+$ ]]; then
    echo "Error: --duration must be a positive integer."
    exit 1
fi

# Auto-prepend vss_ if not present
if [[ ! "$SERIAL" =~ ^(vss_|srv_) ]]; then
    echo "Note: Prepending 'vss_' prefix to serial '$SERIAL'."
    SERIAL="vss_${SERIAL}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BAG_NAME="${SERIAL}_calib_${TIMESTAMP}"
CONTAINER_BAG_PATH="/tmp/${BAG_NAME}"
HOST_ARCHIVE="/tmp/${BAG_NAME}.tar.gz"
LOCAL_ARCHIVE="/tmp/${BAG_NAME}.tar.gz"
LOCAL_BAGS_DIR="$(dirname "$0")/bags"

echo "Target:    $TARGET"
echo "Serial:    $SERIAL"
echo "Duration:  ${DURATION}s"
echo "Bag name:  $BAG_NAME"
echo ""

# Find the main vision container for this serial (exclude diagnostics/hardware sidecars)
echo "Finding container for $SERIAL..."
CONTAINER=$(ssh "$TARGET" "docker ps --filter name=${SERIAL} --format '{{.Names}}' | grep -v -E 'diagnostics|hardware|node-exporter|promtail' | head -1")

if [ -z "$CONTAINER" ]; then
    echo "Error: No running container found matching serial '$SERIAL'."
    echo "   Check running containers with: ssh $TARGET docker ps"
    exit 1
fi

echo "   Found container: $CONTAINER"
echo ""

# Record the rosbag inside the container
echo "Recording ${DURATION}s rosbag (MCAP) inside container..."
ssh "$TARGET" "docker exec '${CONTAINER}' bash -c '
    ROS_DISTRO=\$(ls /opt/ros/ | head -1)
    source /opt/ros/\${ROS_DISTRO}/setup.bash 2>/dev/null || true
    source /ros_ws/install/setup.bash 2>/dev/null || true

    if ! ros2 bag record --storage mcap --help > /dev/null 2>&1; then
        echo \"Installing MCAP storage plugin...\"
        apt-get install -y ros-\${ROS_DISTRO}-rosbag2-storage-mcap
    fi

    rm -rf \"${CONTAINER_BAG_PATH}\"
    timeout ${DURATION} ros2 bag record --storage mcap -o \"${CONTAINER_BAG_PATH}\" \
        /zoom_camera/main \
        /context_camera/main \
        /zoom_camera/camera_info \
        /context_camera/camera_info \
        /tf \
        /tf_static \
        /livox/lidar || true
    echo \"Recording finished.\"
'"
echo "   Recording complete."
echo ""

# Copy bag directory from container to the host's /tmp
echo "Copying bag from container to host /tmp..."
ssh "$TARGET" "
    docker cp '${CONTAINER}:${CONTAINER_BAG_PATH}' /tmp/
    tar czf '${HOST_ARCHIVE}' -C /tmp '${BAG_NAME}'
    echo '   Archived to ${HOST_ARCHIVE}'
"

# Copy the archive from the remote host to local
echo "Copying archive from $TARGET to local..."
scp "$TARGET:${HOST_ARCHIVE}" "${LOCAL_ARCHIVE}"

# Clean up temp files on the remote host
echo "Cleaning up remote temp files..."
ssh "$TARGET" "rm -rf '/tmp/${BAG_NAME}' '${HOST_ARCHIVE}'"

# Extract into bags/
echo "Extracting into bags/..."
mkdir -p "${LOCAL_BAGS_DIR}"
tar xzf "${LOCAL_ARCHIVE}" -C "${LOCAL_BAGS_DIR}"
rm -f "${LOCAL_ARCHIVE}"

echo ""
echo "Done."
echo "   Bag: ${LOCAL_BAGS_DIR}/${BAG_NAME}"
