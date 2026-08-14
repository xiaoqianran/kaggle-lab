# apps — 给人用的界面

实验室的 CLI 在 `labs/`。这里只登记 **产品入口**，避免「网页藏在 015 文件夹里」。

| 产品 | 代码 | 怎么开 |
|------|------|--------|
| 双智对谈 Web | [`labs/015-dual-agent-chat/web`](../labs/015-dual-agent-chat/web/) | Pages：https://xiaoqianran.github.io/kaggle-lab/ · 本地：`python -m kaggle_lab gateway` |

以后若再做独立 UI，在本目录加文件夹，并在 `catalog.py` 把对应 lab 标 `product=True`。
