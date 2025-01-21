python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o state \
  --save-traj --num-procs 10

#python -m mani_skill.trajectory.replay_trajectory \
#  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
#  --use-first-env-state -c pd_joint_delta_pos -o rgbd \
#  --save-traj --num-procs 10
  
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o pointcloud \
  --save-traj --num-procs 10
  
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o rgb+depth+segmentation \
  --save-traj --num-procs 10
  
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o sensor_data \
  --save-traj --num-procs 10
