# generate_env.sh
#!/bin/bash

> .env

(
    . ../real_env/venv/bin/activate
    source /opt/ros/humble/setup.bash
    env
) | grep -E '^(PATH|LD_LIBRARY_PATH|PYTHONPATH|ROS_).*' > .env
