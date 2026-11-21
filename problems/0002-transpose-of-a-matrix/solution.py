def transpose_matrix(matrix: list[list[int | float]]) -> list[list[int | float]]:
    if not matrix or not matrix[0]:
        return []
    
    return [list(col) for col in zip(*matrix)]

# Example usage:
a = [[1, 2, 3], [4, 5, 6]]
print(transpose_matrix(a))  # Output: [[1, 4], [2, 5], [3, 6]]
