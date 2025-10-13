import torch
import kornia
import numpy as np

def differentiable_pnp(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    camera_matrix: torch.Tensor,
    num_iterations: int = 100,
    learning_rate: float = 0.01,
    reprojection_sigma: float = 1.0
):
    """
    Solves for camera pose using a differentiable, robust PnP approach.
    This replaces the discrete RANSAC with a soft, weighted optimization.

    Args:
        points_3d (torch.Tensor): 3D object points in the object's coordinate frame.
                                  Shape: (N, 3).
        points_2d (torch.Tensor): Corresponding 2D image points. Shape: (N, 2).
        camera_matrix (torch.Tensor): The camera intrinsics matrix. Shape: (3, 3).
        num_iterations (int): Number of optimization steps.
        learning_rate (float): Learning rate for the Adam optimizer.
        reprojection_sigma (float): A parameter controlling the "softness" of the
                                    outlier rejection. Smaller values are stricter.

    Returns:
        A tuple containing the optimized translation vector (tvec) and rotation
        vector (rvec) as torch tensors.
    """
    assert len(points_3d) == len(points_2d), "Must have the same number of 3D and 2D points."
    device = points_3d.device

    # 1. Initialize pose parameters (translation and rotation)
    # These are the parameters we want to optimize.
    tvec = torch.zeros(1, 3, device=device, requires_grad=True)
    # Initialize rotation as an axis-angle vector (starts with no rotation)
    rvec = torch.zeros(1, 3, device=device, requires_grad=True)

    # 2. Set up the optimizer
    optimizer = torch.optim.Adam([tvec, rvec], lr=learning_rate)
    
    print("Optimizing pose...")
    for i in range(num_iterations):
        optimizer.zero_grad()

        # 3. Project 3D points to 2D using the current pose estimate
        # kornia's `project_points` is fully differentiable.
        # It takes rvec and tvec directly.
        projected_points_2d = kornia.geometry.camera.project_points(
            points_3d.unsqueeze(0), # Add batch dimension
            camera_matrix.unsqueeze(0),
            rvec,
            tvec
        )

        # 4. Calculate reprojection error
        # This is the L2 distance between the observed 2D points and the projected ones.
        error = torch.linalg.norm(points_2d - projected_points_2d, dim=-1)

        # 5. Differentiable RANSAC: Calculate soft "inlier" weights
        # Instead of a hard threshold, we use a robust function (Geman-McClure)
        # to assign weights. High error -> low weight; low error -> high weight.
        # This smoothly down-weights outliers.
        sigma_sq = reprojection_sigma ** 2
        weights = sigma_sq / (sigma_sq + error**2)

        # 6. Calculate the weighted loss
        loss = torch.mean(weights * error)

        # 7. Backpropagate and optimize
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")

    print("Optimization finished.")
    # Detach from the computation graph to return clean tensors
    return tvec.detach(), rvec.detach()


# --- Example Usage ---
if __name__ == "__main__":
    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Define camera intrinsics (a sample camera)
    cam_matrix = torch.tensor([
        [500., 0., 320.], # fx, 0, cx
        [0., 500., 240.], # 0, fy, cy
        [0., 0., 1.]
    ], device=device)

    # 2. Create sample 3D object points (a simple square)
    object_points_3d = torch.tensor([
        [-1., -1., 5.],
        [1., -1., 5.],
        [1., 1., 5.],
        [-1., 1., 5.],
        # Add some outlier points that don't match the 2D points
        [0., 0., 15.],
        [2., 2., 3.]
    ], device=device)
    
    num_points = len(object_points_3d)

    # 3. Define the ground-truth pose to generate 2D points
    true_rvec = torch.tensor([[0.1, -0.2, 0.05]], device=device)
    true_tvec = torch.tensor([[0.5, 0.3, -0.2]], device=device)

    # 4. Generate the corresponding 2D image points by projecting the 3D points
    # We add some noise to make it more realistic
    image_points_2d = kornia.geometry.camera.project_points(
        object_points_3d.unsqueeze(0),
        cam_matrix.unsqueeze(0),
        true_rvec,
        true_tvec
    )
    image_points_2d += torch.randn_like(image_points_2d) * 0.5 # Add noise

    # Corrupt the outlier points' 2D locations significantly
    image_points_2d[0, -2, :] = torch.tensor([50., 50.], device=device)
    image_points_2d[0, -1, :] = torch.tensor([600., 400.], device=device)
    image_points_2d = image_points_2d.squeeze(0) # Remove batch dimension

    # 5. Run the differentiable PnP solver
    est_tvec, est_rvec = differentiable_pnp(
        object_points_3d,
        image_points_2d,
        cam_matrix,
        num_iterations=200
    )

    print("\n--- Results ---")
    print(f"True Translation: {true_tvec.cpu().numpy().flatten()}")
    print(f"Est. Translation: {est_tvec.cpu().numpy().flatten()}")
    print(f"True Rotation (rvec): {true_rvec.cpu().numpy().flatten()}")
    print(f"Est. Rotation (rvec): {est_rvec.cpu().numpy().flatten()}")