def compute_error_for_line_given_points(b, m, coordinates):
    totalError = 0
    for i in range(0, len(coordinates)):
        x = coordinates[i][0]
        y = coordinates[i][1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(coordinates))
print(compute_error_for_line_given_points(2, 1, [[3,6],[6,9],[12,18]]))
