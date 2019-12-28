import requests
from lxml import html
import os
import json

# noinspection PyUnresolvedReferences
import pathmagic
from tools.project import proj_dir, use_slurm_system

def get_tree(url, cookies=None):
    #print(requests.get(url, cookies=cookies).text)
    return html.fromstring(requests.get(url, cookies=cookies).text)


def get_pages(tree):
    table_rows = tree.xpath('//*[@id="main-leftcontent"]/center/table/tr')

    pages = []

    for r in table_rows:
        cat = r.xpath("td[1]/span/text()")

        if len(cat) > 0:
            cat = cat[0].strip().lower()
            links = r.xpath("td[2]//a/@href")

            pages.append([cat, links])

    return pages


def get_img_links(tree):
    return tree.xpath('//*[@id="main-leftcontent"]/center/table//img/@src')


def download(path, link):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    cont = requests.get(link)
    with open(path, "wb") as f:
        f.write(cont.content)


def main():
    use_slurm_system()

    url = "http://www.grsites.com"
    url_tex = url + "/archive/textures"

    home = get_tree(url_tex)
    pages = get_pages(home)

    global_idx = 0

    for category, links in pages:
        print(category)

        for page_id, link in enumerate(links):
            img_links = get_img_links(get_tree(url + link))

            for i, img_link in enumerate(img_links):
                filepath = proj_dir("crawl", "grsites", category, img_link.split("/")[-1])
                download(filepath, img_link)
                print(global_idx, category, page_id+1, "/", len(links), len(img_links), img_link, ">", filepath)
                global_idx += 1
                #break

            #break

        #break

    #print(json.dumps(page_dict, indent=4))


if __name__ == "__main__":
    main()
