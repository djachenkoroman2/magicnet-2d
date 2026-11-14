# MMDetection Config References

Проект использует ссылки вида `mmdet::<relative/path/to/config.py>`.

Пример:

```toml
[mmdet]
config = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"
```

Во время выполнения ссылка будет разрешена в установленный каталог конфигов MMDetection:

```text
<site-packages>/mmdet/.mim/configs/
```

Это упрощает поддержку шаблона и позволяет быстро переключаться между upstream-конфигами.
