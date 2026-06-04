# Product Images

Place product images here to be served at `/static/products/<filename>`.

## Expected files

| File | Product |
|------|---------|
| `xlc.jpg` | Valvola a saracinesca XLC |
| `argo.jpg` | Valvola a farfalla ARGO |
| `italica353.jpg` | Valvola a sfera ITALICA 353 |
| `protector.jpg` | Valvola di ritegno PROTECTOR |
| `dedalo.jpg` | Giunto DEDALO |
| `vortice.jpg` | Idrante VORTICE |
| `orbis.jpg` | Valvola ORBIS |
| `isis.jpg` | Valvola ISIS |

## How to add/update images

1. Download or export the product image from csasrl.it (wp-admin media library or FTP).
2. Rename the file according to the table above (JPG or PNG both work — update the
   URL in `api/product_images.py` if you use `.png`).
3. Place the file in this directory.
4. Redeploy the application (Render picks up new static files automatically on next deploy).

## Alternative: hotlink from csasrl.it

Instead of self-hosting you can replace the `/static/products/...` URL in
`api/product_images.py` with the full `https://www.csasrl.it/wp-content/uploads/...`
URL.  The widget hides images that return a 404 or fail to load, so broken links
are safe.
