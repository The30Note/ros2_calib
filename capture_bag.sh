#!/bin/bash
# Captures a 5-second rosbag from a running sensor suite container and copies it
# into bags/. Works against a remote server over SSH, or against the local machine
# with --local (no SSH; uses the local Docker daemon directly).
#
# Usage:
#   ./capture_bag.sh user@server --serial vss_016
#   ./capture_bag.sh dockware@beyonce --serial vss_016
#   ./capture_bag.sh dockware@beyonce --serial vss_016 --duration 30
#   ./capture_bag.sh --local --serial vss_016
#   ./capture_bag.sh --local --serial vss_016 --duration 30
#
# Options:
#   --local                Capture from the local Docker daemon instead of SSHing remotely.
#   --serial <serial>      Sensor suite serial (e.g. vss_016). vss_ prefix auto-added if missing.
#   --duration <seconds>   Recording duration in seconds (default: 5)

set -e

TARGET=""
SERIAL=""
DURATION=5
LOCAL=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL=1
            ;;
        --serial)
            SERIAL="$2"
            shift
            ;;
        --duration)
            DURATION="$2"
            shift
            ;;
        --*)
            echo "Unknown parameter: $1"
            exit 1
            ;;
        *)
            if [ -z "$TARGET" ]; then
                TARGET="$1"
            else
                echo "Unexpected argument: $1"
                exit 1
            fi
            ;;
    esac
    shift
done

if [ "$LOCAL" -eq 0 ] && [ -z "$TARGET" ]; then
    echo "Error: Missing target (user@server). Pass a target or use --local."
    echo "Usage: $0 user@server --serial <serial> [--duration <seconds>]"
    echo "       $0 --local --serial <serial> [--duration <seconds>]"
    exit 1
fi

if [ "$LOCAL" -eq 1 ] && [ -n "$TARGET" ]; then
    echo "Error: Don't pass a target (user@server) together with --local."
    exit 1
fi

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

# Run a command either on the remote host (over SSH) or locally, depending on --local.
run_host() {
    if [ "$LOCAL" -eq 1 ]; then
        bash -c "$1"
    else
        ssh "$TARGET" "$1"
    fi
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BAG_NAME="${SERIAL}_calib_${TIMESTAMP}"
CONTAINER_BAG_PATH="/tmp/${BAG_NAME}"
HOST_ARCHIVE="/tmp/${BAG_NAME}.tar.gz"
LOCAL_ARCHIVE="/tmp/${BAG_NAME}.tar.gz"
LOCAL_BAGS_DIR="$(dirname "$0")/bags"

if [ "$LOCAL" -eq 1 ]; then
    echo "Target:    local"
else
    echo "Target:    $TARGET"
fi
echo "Serial:    $SERIAL"
echo "Duration:  ${DURATION}s"
echo "Bag name:  $BAG_NAME"
echo ""

# Find the main vision container for this serial (exclude diagnostics/hardware sidecars)
echo "Finding container for $SERIAL..."
CONTAINER=$(run_host "docker ps --filter name=${SERIAL} --format '{{.Names}}' | grep -v -E 'diagnostics|hardware|node-exporter|promtail' | head -1")

if [ -z "$CONTAINER" ]; then
    echo "Error: No running container found matching serial '$SERIAL'."
    if [ "$LOCAL" -eq 1 ]; then
        echo "   Check running containers with: docker ps"
    else
        echo "   Check running containers with: ssh $TARGET docker ps"
    fi
    exit 1
fi

echo "   Found container: $CONTAINER"
echo ""

# Record the rosbag inside the container
echo "Recording ${DURATION}s rosbag (MCAP) inside container..."
run_host "docker exec '${CONTAINER}' bash -c '
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

mkdir -p "${LOCAL_BAGS_DIR}"

if [ "$LOCAL" -eq 1 ]; then
    # Local: copy the bag straight from the container into bags/.
    echo "Copying bag from container into bags/..."
    rm -rf "${LOCAL_BAGS_DIR}/${BAG_NAME}"
    docker cp "${CONTAINER}:${CONTAINER_BAG_PATH}" "${LOCAL_BAGS_DIR}/"
else
    # Remote: copy bag out of the container, archive it, pull it over SSH, then extract.
    echo "Copying bag from container to host /tmp..."
    ssh "$TARGET" "
        docker cp '${CONTAINER}:${CONTAINER_BAG_PATH}' /tmp/
        tar czf '${HOST_ARCHIVE}' -C /tmp '${BAG_NAME}'
        echo '   Archived to ${HOST_ARCHIVE}'
    "

    echo "Copying archive from $TARGET to local..."
    scp "$TARGET:${HOST_ARCHIVE}" "${LOCAL_ARCHIVE}"

    echo "Cleaning up remote temp files..."
    ssh "$TARGET" "rm -rf '/tmp/${BAG_NAME}' '${HOST_ARCHIVE}'"

    echo "Extracting into bags/..."
    tar xzf "${LOCAL_ARCHIVE}" -C "${LOCAL_BAGS_DIR}"
    rm -f "${LOCAL_ARCHIVE}"
fi

# Clean up the container's temp bag (local docker cp leaves the source in place).
if [ "$LOCAL" -eq 1 ]; then
    docker exec "${CONTAINER}" rm -rf "${CONTAINER_BAG_PATH}" 2>/dev/null || true
fi

echo ""
echo "Done."
echo "   Bag: ${LOCAL_BAGS_DIR}/${BAG_NAME}"
