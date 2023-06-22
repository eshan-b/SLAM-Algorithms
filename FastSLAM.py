import numpy as np
import matplotlib.pyplot as plt

# Robot motion model parameters
alpha = np.array([0.1, 0.01])  # Motion noise parameters

# Landmark measurement model parameters
sigma_r = 0.1  # Range measurement noise
sigma_phi = 0.05  # Bearing measurement noise


def initialize_particles(num_particles):
    # Initialize particles randomly across the space
    particles = []
    for _ in range(num_particles):
        particle = {
            'pose': np.zeros(3),  # (x, y, theta)
            'weight': 1.0 / num_particles,
            'landmarks': {}
        }
        particles.append(particle)
    return particles


def sample_motion_model(particle, control):
    # Sample motion model to update particle's pose
    delta_rot1 = control[0] + np.random.randn() * alpha[0]
    delta_trans = control[1] + np.random.randn() * alpha[1]
    delta_rot2 = control[2] + np.random.randn() * alpha[0]

    theta = particle['pose'][2]
    particle['pose'][0] += delta_trans * np.cos(theta + delta_rot1)
    particle['pose'][1] += delta_trans * np.sin(theta + delta_rot1)
    particle['pose'][2] += delta_rot1 + delta_rot2

    return particle


def measurement_likelihood(particle, measurement):
    likelihood = 1.0
    for landmark_id, (landmark_range, landmark_bearing) in measurement.items():
        if landmark_id not in particle['landmarks']:
            continue

        landmark = particle['landmarks'][landmark_id]
        expected_range = np.sqrt(
            (landmark[0] - particle['pose'][0])**2 + (landmark[1] - particle['pose'][1])**2)
        expected_bearing = np.arctan2(
            landmark[1] - particle['pose'][1], landmark[0] - particle['pose'][0]) - particle['pose'][2]

        range_error = landmark_range - expected_range
        bearing_error = wrap_to_pi(landmark_bearing - expected_bearing)

        range_likelihood = gaussian_likelihood(range_error, sigma_r)
        bearing_likelihood = gaussian_likelihood(bearing_error, sigma_phi)

        likelihood *= range_likelihood * bearing_likelihood

    return likelihood


def resample(particles):
    weights = np.array([particle['weight'] for particle in particles])
    weights /= np.sum(weights)  # Normalize the weights
    indices = np.random.choice(len(particles), size=len(
        particles), replace=True, p=weights)
    resampled_particles = [particles[i].copy() for i in indices]
    for particle in resampled_particles:
        particle['weight'] = 1.0 / len(resampled_particles)
    return resampled_particles


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def gaussian_likelihood(error, sigma):
    return np.exp(-0.5 * (error**2) / (sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def simulate_landmark(pose, landmark):
    x = pose[0] + landmark[0] * np.cos(pose[2] + landmark[1])
    y = pose[1] + landmark[0] * np.sin(pose[2] + landmark[1])
    return x, y


def fastslam(particles, controls, measurements):
    for control, measurement in zip(controls, measurements):
        # Motion update (particle filter)
        for particle in particles:
            particle = sample_motion_model(particle, control)

        # Measurement update (particle filter)
        for particle in particles:
            particle['weight'] *= measurement_likelihood(particle, measurement)

        # Resampling
        particles = resample(particles)

        # Landmark update
        for particle, measurement in zip(particles, measurements):
            for landmark_id, (landmark_range, landmark_bearing) in measurement.items():
                if landmark_id not in particle['landmarks']:
                    particle['landmarks'][landmark_id] = (
                        landmark_range, landmark_bearing)

    return particles


def main():
    num_particles = 100
    particles = initialize_particles(num_particles)

    # Simulated robot controls (motion commands)
    controls = [(0.5, 0.2, 0.1), (0.2, 0.1, -0.3), (0.3, 0.4, 0.2)]

    # Simulated landmark measurements
    measurements = [
        {'landmark1': (1.2, 0.1), 'landmark2': (0.8, -0.3)},
        {'landmark1': (1.1, -0.2)},
        {'landmark2': (0.9, 0.4)}
    ]

    # Run FastSLAM algorithm
    particles = fastslam(particles, controls, measurements)

    # Visualize particles
    plt.figure()
    for particle in particles:
        x, y, theta = particle['pose']
        plt.plot(x, y, 'bo')
        for landmark_id, landmark in particle['landmarks'].items():
            landmark_x, landmark_y = simulate_landmark(
                particle['pose'], landmark)
            plt.plot(landmark_x, landmark_y, 'rx')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('FastSLAM')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    main()
