import base64
import os.path
import cv2
import numpy as np

__all__ = [
    'get_grid_shape', 'get_blank_image', 'load_image', 'save_image',
    'resize_image', 'add_text_to_image', 'fuse_images', 'HtmlPageVisualizer',
    'VideoReader', 'VideoWriter'
]


def get_grid_shape(size, row=0, col=0, is_portrait=False):
  assert isinstance(size, int)
  assert isinstance(row, int)
  assert isinstance(col, int)
  if size == 0:
    return (0, 0)

  if row > 0 and col > 0 and row * col != size:
    row = 0
    col = 0

  if row > 0 and size % row == 0:
    return (row, size // row)
  if col > 0 and size % col == 0:
    return (size // col, col)

  row = int(np.sqrt(size))
  while row > 0:
    if size % row == 0:
      col = size // row
      break
    row = row - 1

  return (col, row) if is_portrait else (row, col)


def get_blank_image(height, width, channels=3, is_black=True):
  shape = (height, width, channels)
  if is_black:
    return np.zeros(shape, dtype=np.uint8)
  return np.ones(shape, dtype=np.uint8) * 255


def load_image(path):
  if not os.path.isfile(path):
    return None

  image = cv2.imread(path)
  return image[:, :, ::-1]


def save_image(path, image):
  if image is None:
    return

  assert len(image.shape) == 3 and image.shape[2] in [1, 3]
  cv2.imwrite(path, image[:, :, ::-1])


def resize_image(image, *args, **kwargs):
  if image is None:
    return None

  assert image.ndim == 3 and image.shape[2] in [1, 3]
  image = cv2.resize(image, *args, **kwargs)
  if image.ndim == 2:
    return image[:, :, np.newaxis]
  return image


def add_text_to_image(image,
                      text='',
                      position=None,
                      font=cv2.FONT_HERSHEY_TRIPLEX,
                      font_size=1.0,
                      line_type=cv2.LINE_8,
                      line_width=1,
                      color=(255, 255, 255)):
  if image is None or not text:
    return image

  cv2.putText(img=image,
              text=text,
              org=position,
              fontFace=font,
              fontScale=font_size,
              color=color,
              thickness=line_width,
              lineType=line_type,
              bottomLeftOrigin=False)

  return image


def fuse_images(images,
                image_size=None,
                row=0,
                col=0,
                is_row_major=True,
                is_portrait=False,
                row_spacing=0,
                col_spacing=0,
                border_left=0,
                border_right=0,
                border_top=0,
                border_bottom=0,
                black_background=True):
  if images is None:
    return images

  if not images.ndim == 4:
    raise ValueError(f'Input `images` should be with shape [num, height, '
                     f'width, channels], but {images.shape} is received!')

  num, image_height, image_width, channels = images.shape
  if image_size is not None:
    if isinstance(image_size, int):
      image_size = (image_size, image_size)
    assert isinstance(image_size, (list, tuple)) and len(image_size) == 2
    width, height = image_size
  else:
    height, width = image_height, image_width
  row, col = get_grid_shape(num, row=row, col=col, is_portrait=is_portrait)
  fused_height = (
      height * row + row_spacing * (row - 1) + border_top + border_bottom)
  fused_width = (
      width * col + col_spacing * (col - 1) + border_left + border_right)
  fused_image = get_blank_image(
      fused_height, fused_width, channels=channels, is_black=black_background)
  images = images.reshape(row, col, image_height, image_width, channels)
  if not is_row_major:
    images = images.transpose(1, 0, 2, 3, 4)

  for i in range(row):
    y = border_top + i * (height + row_spacing)
    for j in range(col):
      x = border_left + j * (width + col_spacing)
      if image_size is not None:
        image = cv2.resize(images[i, j], image_size)
      else:
        image = images[i, j]
      fused_image[y:y + height, x:x + width] = image

  return fused_image


def get_sortable_html_header(column_name_list, sort_by_ascending=False):
  header = '\n'.join([
      '<script type="text/javascript">',
      'var column_idx;',
      'var sort_by_ascending = ' + str(sort_by_ascending).lower() + ';',
      '',
      'function sorting(tbody, column_idx){',
      '  this.column_idx = column_idx;',
      '  Array.from(tbody.rows)',
      '       .sort(compareCells)',
      '       .forEach(function(row) { tbody.appendChild(row); })',
      '  sort_by_ascending = !sort_by_ascending;',
      '}',
      '',
      'function compareCells(row_a, row_b) {',
      '  var val_a = row_a.cells[column_idx].innerText;',
      '  var val_b = row_b.cells[column_idx].innerText;',
      '  var flag = sort_by_ascending ? 1 : -1;',
      '  return flag * (val_a > val_b ? 1 : -1);',
      '}',
      '</script>',
      '',
      '<html>',
      '',
      '<head>',
      '<style>',
      '  table {',
      '    border-spacing: 0;',
      '    border: 1px solid black;',
      '  }',
      '  th {',
      '    cursor: pointer;',
      '  }',
      '  th, td {',
      '    text-align: left;',
      '    vertical-align: middle;',
      '    border-collapse: collapse;',
      '    border: 0.5px solid black;',
      '    padding: 8px;',
      '  }',
      '  tr:nth-child(even) {',
      '    background-color: #d2d2d2;',
      '  }',
      '</style>',
      '</head>',
      '',
      '<body>',
      '',
      '<table>',
      '<thead>',
      '<tr>',
      ''])
  for idx, column_name in enumerate(column_name_list):
    header += f'  <th onclick="sorting(tbody, {idx})">{column_name}</th>\n'
  header += '</tr>\n'
  header += '</thead>\n'
  header += '<tbody id="tbody">\n'

  return header


def get_sortable_html_footer():
  return '</tbody>\n</table>\n\n</body>\n</html>\n'


def encode_image_to_html_str(image, image_size=None):
  if image is None:
    return ''

  assert len(image.shape) == 3 and image.shape[2] in [1, 3]

  # Change channel order to `BGR`, which is opencv-friendly.
  image = image[:, :, ::-1]

  # Resize the image if needed.
  if image_size is not None:
    if isinstance(image_size, int):
      image_size = (image_size, image_size)
    assert isinstance(image_size, (list, tuple)) and len(image_size) == 2
    image = cv2.resize(image, image_size)

  # Encode the image to html-format string.
  encoded_image = cv2.imencode(".jpg", image)[1].tostring()
  encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
  html_str = f'<img src="data:image/jpeg;base64, {encoded_image_base64}"/>'

  return html_str


class HtmlPageVisualizer(object):
  def __init__(self,
               num_rows=0,
               num_cols=0,
               grid_size=0,
               is_portrait=False,
               viz_size=None):
    if grid_size > 0:
      num_rows, num_cols = get_grid_shape(
          grid_size, row=num_rows, col=num_cols, is_portrait=is_portrait)
    assert num_rows > 0 and num_cols > 0

    self.num_rows = num_rows
    self.num_cols = num_cols
    self.viz_size = viz_size
    self.headers = ['' for _ in range(self.num_cols)]
    self.cells = [[{
        'text': '',
        'image': '',
    } for _ in range(self.num_cols)] for _ in range(self.num_rows)]

  def set_header(self, column_idx, content):
    self.headers[column_idx] = content

  def set_headers(self, contents):
    if isinstance(contents, str):
      contents = [contents]
    assert isinstance(contents, (list, tuple))
    assert len(contents) == self.num_cols
    for column_idx, content in enumerate(contents):
      self.set_header(column_idx, content)

  def set_cell(self, row_idx, column_idx, text='', image=None):
    self.cells[row_idx][column_idx]['text'] = text
    self.cells[row_idx][column_idx]['image'] = encode_image_to_html_str(
        image, self.viz_size)

  def save(self, save_path):
    html = ''
    for i in range(self.num_rows):
      html += f'<tr>\n'
      for j in range(self.num_cols):
        text = self.cells[i][j]['text']
        image = self.cells[i][j]['image']
        if text:
          html += f'  <td>{text}<br><br>{image}</td>\n'
        else:
          html += f'  <td>{image}</td>\n'
      html += f'</tr>\n'

    header = get_sortable_html_header(self.headers)
    footer = get_sortable_html_footer()

    with open(save_path, 'w') as f:
      f.write(header + html + footer)


class VideoReader(object):
  def __init__(self, path):
    if not os.path.isfile(path):
      raise ValueError(f'Video `{path}` does not exist!')

    self.path = path
    self.video = cv2.VideoCapture(path)
    assert self.video.isOpened()
    self.position = 0

    self.length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
    self.frame_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.frame_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.fps = self.video.get(cv2.CAP_PROP_FPS)

  def __del__(self):
    self.video.release()

  def read(self, position=None):
    if position is not None and position < self.length:
      self.video.set(cv2.CAP_PROP_POS_FRAMES, position)
      self.position = position

    success, frame = self.video.read()
    self.position = self.position + 1

    return frame[:, :, ::-1] if success else None


class VideoWriter(object):
  def __init__(self, path, frame_height, frame_width, fps=24, codec='DIVX'):
    self.path = path
    self.frame_height = frame_height
    self.frame_width = frame_width
    self.fps = fps
    self.codec = codec

    self.video = cv2.VideoWriter(filename=path,
                                 fourcc=cv2.VideoWriter_fourcc(*codec),
                                 fps=fps,
                                 frameSize=(frame_width, frame_height))

  def __del__(self):
    self.video.release()

  def write(self, frame):
    self.video.write(frame[:, :, ::-1])
