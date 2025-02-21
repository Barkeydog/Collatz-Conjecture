def test_loop(num):
    print(num)

    original_number = num
    num = ((9 * num + 3) / 2) + 1
    
    while num != original_number:
        num = num / 2
        if num % 1 != 0:
            return
    
    print(f"Success! {original_number} is the result.")

# Testing odd numbers greater than 3, starting at 3 and incrementing by 2
num = 3
while True:
    test_loop(num)
    num += 2
