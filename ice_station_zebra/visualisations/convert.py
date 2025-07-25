import io
from matplotlib.figure import Figure
from PIL import Image, ImageFile

def image_from_figure(figure: Figure) -> ImageFile:
    """Convert a matplotlib Figure to a PIL Image."""
    img_buf = io.BytesIO()
    figure.savefig(img_buf, format='png')
    img_buf.seek(0)
    return Image.open(img_buf)