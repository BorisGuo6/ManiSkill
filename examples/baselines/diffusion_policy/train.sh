python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgbd.pd_joint_delta_pos.cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 

python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.pointcloud.pd_joint_delta_pos.cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 
  
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb+depth+segmentation.pd_joint_delta_pos.cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 
  
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.sensor_data.pd_joint_delta_pos.cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 
