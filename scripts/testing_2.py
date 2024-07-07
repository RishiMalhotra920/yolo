# import torch

# # Example usage
# image_width = 448
# image_height = 448
# bboxes = torch.tensor(
#     [
#         [100, 100, 200, 200, 5],  # cell 2, 2
#         [101, 100, 200, 200, 5],
#         [110, 110, 200, 200, 4],
#         [110, 110, 200, 200, 4],
#         [150, 150, 250, 250, 5],
#         [300, 300, 400, 400, 10],
#         # Add more bounding boxes as needed
#     ]
# )

# result = categorize_bboxes(bboxes, image_width, image_height)
# # print(result.shape)  # Should be torch.Size([7, 7, 30])

# for grid_x in range(0, 7):
#     for grid_y in range(0, 7):
#         print(f"Cell ({grid_x}, {grid_y}):")
#         print(result[grid_x, grid_y, :])
