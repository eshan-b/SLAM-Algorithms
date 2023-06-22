import numpy as np
import g2o

# Generate some synthetic data
num_poses = 10
num_landmarks = 5

# Generate random poses
true_poses = [np.array([np.random.randn(), np.random.randn(),
                       np.random.randn()]) for _ in range(num_poses)]

# Generate random landmarks
true_landmarks = [np.random.randn(2) for _ in range(num_landmarks)]

# Generate random odometry measurements
odom_measurements = [true_poses[i+1] - true_poses[i]
                     for i in range(num_poses-1)]

# Generate random landmark measurements
landmark_measurements = []
for i in range(num_poses):
    measurements = {}
    for j in range(num_landmarks):
        landmark_pose = np.append(true_landmarks[j], 0)
        pose_diff = true_poses[i] - landmark_pose
        distance = np.linalg.norm(pose_diff[:2])
        angle = np.arctan2(pose_diff[1], pose_diff[0]) - true_poses[i][2]
        measurements[j] = (distance + np.random.randn() *
                           0.1, angle + np.random.randn() * 0.1)
    landmark_measurements.append(measurements)

# Create a g2o optimizer
optimizer = g2o.SparseOptimizer()

# Add the vertices for poses
for i in range(num_poses):
    pose = g2o.SE2(true_poses[i][0], true_poses[i][1], true_poses[i][2])
    vertex_pose = g2o.VertexSE2()
    vertex_pose.set_id(i)
    vertex_pose.set_estimate(pose)
    vertex_pose.set_fixed(i == 0)  # Fix the first pose
    optimizer.add_vertex(vertex_pose)

# Add the vertices for landmarks
for i in range(num_landmarks):
    vertex_landmark = g2o.VertexPointXY()
    vertex_landmark.set_id(i + num_poses)
    vertex_landmark.set_estimate(true_landmarks[i])
    optimizer.add_vertex(vertex_landmark)

# Add the odometry edges
for i in range(num_poses - 1):
    odometry_edge = g2o.EdgeSE2()
    odometry_edge.set_vertex(0, optimizer.vertex(i))
    odometry_edge.set_vertex(1, optimizer.vertex(i + 1))
    odometry_edge.set_measurement(g2o.SE2(
        odom_measurements[i][0], odom_measurements[i][1], odom_measurements[i][2]))
    odometry_edge.set_information(np.eye(3))
    optimizer.add_edge(odometry_edge)

# Add the landmark measurement edges
for i in range(num_poses):
    measurements = landmark_measurements[i]
    for j, measurement in measurements.items():
        landmark_edge = g2o.EdgeSE2PointXY()
        landmark_edge.set_vertex(0, optimizer.vertex(i))
        landmark_edge.set_vertex(1, optimizer.vertex(j + num_poses))
        landmark_edge.set_measurement(np.array(measurement))
        landmark_edge.set_information(np.eye(2))
        optimizer.add_edge(landmark_edge)

# Optimize the graph
optimizer.initialize_optimization()
optimizer.optimize(10)

# Retrieve the optimized poses and landmarks
optimized_poses = [optimizer.vertex(i).estimate().to_vector()[
    :3] for i in range(num_poses)]
optimized_landmarks = [optimizer.vertex(
    i + num_poses).estimate() for i in range(num_landmarks)]

# Print the results
print("True Poses:")
print(true_poses)
print("Optimized Poses:")
print(optimized_poses)
print("True Landmarks:")
print(true_landmarks)
print("Optimized Landmarks:")
print(optimized_landmarks)
