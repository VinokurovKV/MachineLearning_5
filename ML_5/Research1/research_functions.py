def are_multisets_equal(x, y):
    x.sort()
    y.sort()

    if x == y:
        return True
    else:
        return False
    
def max_prod_mod_3(x):
    max_product = -1
    for i in range(len(x) - 1):
        current_product = x[i] * x[i + 1]
        if x[i] % 3 == 0 or x[i + 1] % 3 == 0:
            if max_product == -1 or current_product > max_product:
                max_product = current_product
    
    return max_product

def convert_image(image, weights):
    height = len(image)
    width = len(image[0])
    num_channels = len(weights)
    
    result = []
    
    for i in range(height):
        row_result = []
        for j in range(width):
            pixel_sum = 0
            for k in range(num_channels):
                pixel_sum += image[i][j][k] * weights[k]
            row_result.append(pixel_sum)
        result.append(row_result)
    
    return result

def rle_scalar(x, y):
    total_x = sum(count for _, count in x)
    total_y = sum(count for _, count in y)
    
    if total_x != total_y:
        return -1
    
    decoded_x = []
    for value, count in x:
        decoded_x.extend([value] * count)
    
    decoded_y = []
    for value, count in y:
        decoded_y.extend([value] * count)
    
    result = 0
    for i in range(total_x):
        result += decoded_x[i] * decoded_y[i]
    
    return result

def cosine_distance(X, Y):
    ans = []
    for i in range(len(X)):
        ans.append([])
        x_normalized=sum(tmp**2 for tmp in X[i])**0.5
        for j in range(len(Y)):
            y_normalized = sum(tmp**2 for tmp in Y[j])**0.5
            if x_normalized==0 or y_normalized==0:
                ans[i].append(float(1))
            else:
                ans[i].append(sum(X[i][k] * Y[j][k] for k in range(len(X[0])))/(x_normalized*y_normalized))
    return ans