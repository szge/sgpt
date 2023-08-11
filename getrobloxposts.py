"""Create a list of Roblox Forum Archive posts from a user's profile."""

import requests
from bs4 import BeautifulSoup


def get_roblox_posts():
    """fetch from https://archive.froast.io/users/294568/asc"""
    page = requests.get("https://archive.froast.io/users/294568/asc")
    soup = BeautifulSoup(page.content, "html.parser")
    posts = []
    next_button = soup.find("a", class_="Forward Link-Enabled")
    # turn this into a loop until there are no more pages
    while next_button is not None:
        # get all posts
        for tr in soup.find_all("tr")[1::2]:
            topic_element = tr.find("div", class_="Topic")
            body_element = tr.find("div", class_="Body")

            if topic_element is not None and body_element is not None:
                topic = topic_element.text
                body = body_element.text
                posts.append(topic + "\n\ncntkillme:\n\n" + body + "\n\n")

        # get next page
        next_button = soup.find("a", class_="Forward Link-Enabled")
        if next_button is not None and len(posts) < 4000:
            url = "https://archive.froast.io" + next_button["href"]
            print(url)
            page = requests.get(url)
            soup = BeautifulSoup(page.content, "html.parser")
        else:
            break

    # write to file
    with open("corpora/cntkillme.txt", "w", encoding="utf-8") as f:
        f.writelines(posts)


if __name__ == "__main__":
    get_roblox_posts()
