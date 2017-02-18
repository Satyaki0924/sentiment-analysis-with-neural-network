"""
This is a project by Satyaki Sanyal.
This project must be used for educational purposes only.

Follow me on:
LinkedIn - https://www.linkedin.com/in/satyaki-sanyal-708424b7/
Github - https://github.com/Satyaki0924/
Researchgate - https://www.researchgate.net/profile/Satyaki_Sanyal
"""
from functions.execute import SentimentNetwork


def main():
    while True:
        try:
            print("Enter: 1. to test accuracy || 2. analyse your data")
            inp = input('>>\t')
            if int(inp) == 1 or int(inp) == 2:
                SentimentNetwork(int(inp))
                break
            else:
                print(str(inp) + " is not accepted. Try again...")
        except:
            pass


if __name__ == '__main__':
    main()
