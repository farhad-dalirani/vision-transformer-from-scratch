import torch
import torch.nn.functional as F


def visualize_positional_embeddings_norm_map(
    positional_embedding: torch.Tensor,
    grid_width: int,
    grid_height: int,
) -> torch.Tensor:
    """
    Computes a normalized L2-norm heatmap of positional embeddings.

    This function converts positional embeddings into a 2D spatial map
    by reshaping patch tokens into a grid and computing the L2 norm of
    each embedding vector. The resulting norm values are min-max
    normalized to the range [0, 1].

    The CLS token is assumed to be present at index 0 and is removed
    before reshaping.

    Args:
        positional_embedding (torch.Tensor):
            Positional embedding tensor of shape (1, N+1, D), where:
                - N is the number of patch tokens,
                - D is the embedding dimension.
            The first token is assumed to be the CLS token.

        grid_width (int):
            Number of patch tokens along the width of the image grid.

        grid_height (int):
            Number of patch tokens along the height of the image grid.

    Returns:
        torch.Tensor:
            A normalized heatmap tensor of shape (1, grid_height, grid_width),
            where each value represents the L2 norm magnitude of the
            corresponding positional embedding vector.
    """

    # Remove batch dimension
    pos = positional_embedding.squeeze(0)  # (N+1, D)

    # Remove CLS token
    pos = pos[1:]

    # Reshape to grid
    pos = pos.reshape(grid_height, grid_width, -1)  # (H, W, D)

    # Compute L2 norm across embedding dimension
    norm_map = torch.linalg.vector_norm(pos, ord=2, dim=2)  # (H, W)

    # Normalize to [0, 1]
    min_val = norm_map.min()
    max_val = norm_map.max()
    norm_map = (norm_map - min_val) / (max_val - min_val + 1e-8)

    norm_map = norm_map.unsqueeze(0)

    return norm_map


def visualize_positional_embeddings_similarity_map(
    positional_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Computes a pairwise cosine similarity matrix between positional embeddings.

    This function removes the batch dimension and CLS token from the
    positional embedding tensor, normalizes each embedding vector, and
    computes the full pairwise cosine similarity matrix using matrix
    multiplication.

    Args:
        positional_embedding (torch.Tensor):
            Positional embedding tensor of shape (1, N+1, D), where:
                - N is the number of patch tokens,
                - D is the embedding dimension.
            The first token is assumed to be the CLS token.

    Returns:
        torch.Tensor:
            Cosine similarity matrix of shape (1, N, N), where each value
            represents the cosine similarity between two positional
            embedding vectors.
    """
    # Remove batch dimension → (N+1, D)
    pos = positional_embedding.squeeze(0)

    # Remove CLS token → (N, D)
    pos = pos[1:]

    # Normalize embeddings
    pos = F.normalize(pos, p=2, dim=1)

    # Compute pairwise cosine similarity
    cosine_sim = torch.matmul(pos, pos.T)  # (N, N)

    cosine_sim = cosine_sim.unsqueeze(0)

    return cosine_sim


def log_positional_embedding(
    writer,
    model,
    step: int,
) -> None:
    """
    Logs positional embedding visualizations to TensorBoard.

    This function retrieves the model's positional embeddings and logs:
        1. A spatial L2-norm heatmap of positional embeddings.
        2. A pairwise cosine similarity matrix between positional tokens.

    The outputs are written as images under the
    "positional_embeddings/" namespace in TensorBoard.

    Args:
        writer:
            TensorBoard SummaryWriter instance.

        model:
            Vision Transformer model that exposes
            `get_positional_embeddings()` and
            `get_token_grid_size()` methods.

        step (int):
            Global step value used for TensorBoard logging.
    """
    positional_embedding = model.get_positional_embeddings()
    grid_size = model.get_token_grid_size()

    norm_map = visualize_positional_embeddings_norm_map(
        positional_embedding=positional_embedding,
        grid_height=grid_size,
        grid_width=grid_size,
    )

    cosine_similarity_map = visualize_positional_embeddings_similarity_map(
        positional_embedding=positional_embedding
    )

    writer.add_image("positional_embeddings/norm_map", norm_map, global_step=step)
    writer.add_image(
        "positional_embeddings/cosine_similarity",
        cosine_similarity_map,
        global_step=step,
    )
