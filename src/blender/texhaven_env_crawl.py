import requests
from lxml import html
import os


def get_tree(url, cookies=None):
    print(url)
    return html.fromstring(requests.get(url, cookies=cookies).text)


def get_pages(tree):
    return tree.xpath('//*[@id="hdri-grid"]/a/@href')


# map_type = ['diff', 'spec']
def get_map_link(tree, map_type):
    res = tree.xpath(f'//div[div[@map="{map_type}"]]//a/@href')

    return res[0] if len(res) > 0 else None


def download(path, link):
    print(link, ">", path)

    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    cont = requests.get(link)

    if not cont.content.startswith(b"<html>"):
        with open(path, "wb") as f:
            f.write(cont.content)

        return True
    else:
        return False


def main():
    url = "https://hdrihaven.com"
    url_tex = url + "/hdris/category/?c=all"

    home = get_tree(url_tex)
    pages = [url + u for u in get_pages(home)]

    print(len(pages))

    tex_ids = [page.split("?h=")[1] for page in pages]
    print(tex_ids)

    download_path = "/home/kevin/Documents/master-thesis/scene_pool/TextureHavenEnv"

    for tex_id in tex_ids:
        link = f"https://hdrihaven.com/files/hdris/{tex_id}_1k.hdr"
        if not download(os.path.join(download_path, tex_id + "_1k.hdr"), link):
            print(link, "not found")


if __name__ == "__main__":
    main()
