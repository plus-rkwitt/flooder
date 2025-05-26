import torch


def generate_swiss_cheese_points(N, rect_min, rect_max, k, void_radius_range):
    # Generate void centers and radii
    void_centers = []
    void_radii = []
    for _ in range(k):
        while True:
            void_center = (rect_min + void_radius_range[1]) + (
                rect_max - rect_min - 2 * void_radius_range[1]
            ) * torch.rand(1, rect_min.shape[0])
            void_radius = void_radius_range[0] + (
                void_radius_range[1] - void_radius_range[0]
            ) * torch.rand(1)
            is_ok = True
            for i in range(len(void_centers)):
                if (
                    torch.norm(void_center - void_centers[i])
                    < void_radius + void_radii[i]
                ):
                    is_ok = False
            if is_ok:
                void_centers.append(void_center)
                void_radii.append(void_radius)
                break
    void_centers = torch.cat(void_centers)
    void_radii = torch.cat(void_radii)

    points = []
    while len(points) < N:
        # Generate a random point in the rectangular region
        point = rect_min + (rect_max - rect_min) * torch.rand(rect_min.shape[0])

        # Check if the point is inside any void
        distances = torch.norm(point - void_centers, dim=1)
        if not torch.any(distances < void_radii):
            points.append(point)

    return torch.stack(points, dim=0), void_radii


def generate_donut_points(N, center, radius, width):  # 2D
    angles = torch.rand(N) * 2 * torch.pi  # Random angles
    r = (
        radius - width + width * torch.sqrt(torch.rand(N))
    )  # Random radii (sqrt ensures uniform distribution in annulus)
    x = center[0] + r * torch.cos(angles)
    y = center[1] + r * torch.sin(angles)
    return torch.stack((x, y), dim=1)


def generate_noisy_torus_points(
    num_points=1000, R=3, r=1, noise_std=0.02, device="cpu"
):
    # Sample angles
    theta = torch.rand(num_points, device=device) * 2 * torch.pi
    phi = torch.rand(num_points, device=device) * 2 * torch.pi

    # Parametric equations of a torus
    x = (R + r * torch.cos(phi)) * torch.cos(theta)
    y = (R + r * torch.cos(phi)) * torch.sin(theta)
    z = r * torch.sin(phi)

    # Stack into a (num_points, 3) tensor
    points = torch.stack((x, y, z), dim=1)

    # Add Gaussian noise
    noise = torch.randn_like(points) * noise_std
    noisy_points = points + noise

    return noisy_points
