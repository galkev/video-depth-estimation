import requests
from lxml import html
import os


def get_tree(url, cookies=None):
    return html.fromstring(requests.get(url, cookies=cookies).text)


def get_pages(tree):
    return tree.xpath('//*[@id="item-grid"]/a/@href')


# map_type = ['diff', 'spec']
def get_map_link(tree, map_type):
    res = tree.xpath(f'//div[div[@map="{map_type}"]]//a/@href')

    return res[0] if len(res) > 0 else None


def download(path, link):
    print(link, ">", path)

    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    cont = requests.get(link)
    with open(path, "wb") as f:
        f.write(cont.content)


def main():
    url = "https://texturehaven.com"
    url_tex = url + "/textures/"

    home = get_tree(url_tex)
    pages = [url + u for u in get_pages(home)]

    maps = ["diff", "spec"]

    download_path = "/home/kevin/Documents/master-thesis/scene_pool/TextureHaven"

    for page in pages:
        mat_name = page.split("?t=")[1]
        page_tree = get_tree(page)

        for map_type in maps:
            link = get_map_link(page_tree, map_type)

            if link is not None:
                link = url + link
                ext = "." + link.split(".")[-1]
                download(os.path.join(download_path, mat_name, map_type + ext), link)
            else:
                print(page, "no", map_type)

        print(mat_name)

        # break
    #print(pages)


if __name__ == "__main__":
    main()
