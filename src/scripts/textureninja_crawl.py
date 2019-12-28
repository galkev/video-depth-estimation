import time
import json
import os
import requests
import glob


# noinspection PyUnresolvedReferences
import pathmagic
from tools.project import proj_dir, use_slurm_system


tex_ninja_cat = {
    "blood": "https://texture.ninja/textures/Blood/11",
    "brick": "https://texture.ninja/textures/Brick/1",
    "concrete": "https://texture.ninja/category/Concrete/3",
    "covers": "https://texture.ninja/textures/Covers/33",
    "cracks": "https://texture.ninja/textures/Cracks/6",
    "decorative": "https://texture.ninja/textures/Decorative/21",
    "dirt": "https://texture.ninja/category/Dirt/5",
    "doors": "https://texture.ninja/textures/Doors/7",
    "fabric": "https://texture.ninja/textures/Fabric/8",
    "fingerprints": "https://texture.ninja/textures/Fingerprints/12",
    "graffiti": "https://texture.ninja/textures/Graffiti/14",
    "grates": "https://texture.ninja/textures/Grates/32",
    "ground": "https://texture.ninja/category/Ground/10",
    "leaves": "https://texture.ninja/textures/Leaves/4",
    "metal": "https://texture.ninja/category/Metal/41",
    "misc": "https://texture.ninja/category/Misc/18",
    "paint": "https://texture.ninja/category/Paint/22",
    "plaster": "https://texture.ninja/category/Plaster/45",
    "rock": "https://texture.ninja/textures/Rock/13",
    "rust": "https://texture.ninja/category/Rust/35",
    "signs": "https://texture.ninja/category/Signs/15",
    "stone": "https://texture.ninja/category/Stone/17",
    "wood": "https://texture.ninja/category/Wood/2"
}


class TexNinjaCrawler:
    def __init__(self):
        from selenium import webdriver
        from selenium.webdriver.support.wait import WebDriverWait

        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)

    def _get_tex_links(self):
        return [ele.get_attribute("alt") for ele in self.driver.find_elements_by_xpath(
            '//div[@class="ReactVirtualized__Grid ReactVirtualized__List"]//div/img')]

    def get_links(self, url):
        self.driver.get(url)

        all_links = []

        cur_links = self._get_tex_links()
        while len(all_links) == 0 or cur_links[-1] != all_links[-1]:
            print(cur_links)
            all_links.extend(cur_links)

            self.driver.execute_script("window.scrollBy(0, 500);")
            time.sleep(0.05)
            cur_links = self._get_tex_links()

        all_links = list(set(all_links))

        return all_links


def collect():
    crawler = TexNinjaCrawler()

    tex_names = {k: crawler.get_links(v) for k, v in tex_ninja_cat.items()}

    with open("links/tex_ninja_links.json", "w") as f:
        json.dump(tex_names, f)


def check_content(cont):
    return not cont.startswith(b'<?xml')


def download(path, link):
    cont = requests.get(link).content

    if not check_content(cont):
        print(link, "not found")
        return False
    else:
        if not os.path.isdir(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        cont = requests.get(link)
        with open(path, "wb") as f:
            f.write(cont.content)


def download_data(link_file):
    with open(link_file, "r") as f:
        link_dict = json.load(f)

    for k, v in link_dict.items():
        print(k, len(v))

    print("Total tex count", sum(len(links) for links in link_dict.values()))

    base_url = "https://texture-ninja.nyc3.cdn.digitaloceanspaces.com/"

    for category, links in link_dict.items():
        for link in links:
            filepath = proj_dir("crawl", "texninja", category, link)
            download(filepath, base_url + link)

            #break
        #break


    #download("--", base_url + "dsa.jpg")


def main():
    use_slurm_system()

    # collect()
    download_data("links/tex_ninja_links.json")


if __name__ == '__main__':
    main()
