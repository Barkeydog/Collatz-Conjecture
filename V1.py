import time

def intersection_equations():
    x = 1
    z = 2  
    b = 2 ** z
    y1 = x
    y2 = ((3 * x) + 1) / b
    start_time = time.time()  # Record start time

    while True:
        if y2 > y1:
            while y2 > y1:
                z += 1
                b = 2 ** z
                y2 = ((3 * x) + 1) / b  # Recalculate y2
        else:
            y1 = x
            if y1 > 0 and y2 > 0 and y1 % 2 != 0 and y2 % 2 != 0 and y1 == y2:
                print(f"\rIntersection: x = {int(x)}, y = {int(y1)}")
            else:
                print(f"\rCurrent x: {x}   Speed: {x / (time.time() - start_time):.2f} updates/second", end='')  # Use \r to update only x
            x += 1
            z = 1  # Reset z to 1
            b = 2 ** z  # Reset b
            y2 = (3 * (y2 / 2) + 1) / b

# Call the function
intersection_equations()
