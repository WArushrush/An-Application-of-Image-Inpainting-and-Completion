import numpy as np
from PIL import Image
import time
import cv2

global img
global point1, point2
global min_x, min_y, width, height, max_x, max_y


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, min_x, min_y, width, height, max_x, max_y
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 2)
        cv2.imshow('image', img2)
        min_y = min(point1[0], point2[0])
        min_x = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        max_x = min_x + height
        max_y = min_y + width


def overlap_restricted_area(x, y, patch_size, min_x, max_x, min_y, max_y):
    dx0 = dy0 = patch_size // 2
    minx1 = x - dx0
    miny1 = y - dy0
    maxx1 = x + dx0
    maxy1 = y + dy0
    minx2 = min_x
    miny2 = min_y
    maxx2 = max_x
    maxy2 = max_y
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:
        return False
    else:
        return True


def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0] + p_size, a[1]:a[1] + p_size, :]
    patch_b = B[b[0] - p:b[0] + p + 1, b[1] - p:b[1] + p + 1, :]
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist


def cal_alpha(dis, gamma=2.0):
    return gamma ** (-dis)


def reconstruction(f, A, B, p_size, dist, min_x, max_x, min_y, max_y, itter):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    temp = np.zeros_like(A)
    p = p_size // 2
    for i in range(A_h):
        for j in range(A_w):
            cnt = 0
            ans = np.zeros(3)
            for m in range(-p, p + 1, 1):
                for n in range(-p, p + 1, 1):
                    if not ((0 <= i + m < A_h) and (0 <= j + n < A_w)):
                        continue
                    if not ((0 <= f[i + m][j + n][0] - m < B_h) and (0 <= f[i + m][j + n][1] - n < B_w)):
                        continue
                    if overlap_restricted_area(f[i + m][j + n][0] - m, f[i + m][j + n][1] - n, p_size, min_x, max_x,
                                               min_y,
                                               max_y):
                        continue
                    alpha = cal_alpha(dis=dist[i + m, j + n])
                    cnt += alpha
                    ans += alpha * B[f[i + m][j + n][0] - m, f[i + m][j + n][1] - n, :]
            temp[i, j, :] = ans / cnt
    tmp = np.copy(B)
    # temp = cv2.GaussianBlur(temp, (3, 3), 0)
    tmp[min_x:min_x + A_h, min_y:min_y + A_w, :] = temp
    # Image.fromarray(tmp).show()
    return tmp, temp


def initialization(A, B, f, p_size, min_x, max_x, min_y, max_y, create_f=False):
    A_h = np.size(A, 0)
    A_w = np.size(A, 1)
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    # A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    A_padding = B[min_x - p:min_x + A_h + p, min_y - p:min_y + A_w + p, :]
    A_padding[p:A_h + p, p:A_w + p, :] = A
    random_B_r = np.random.randint(p, B_h - p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w - p, [A_h, A_w])
    for i in range(A_h):
        for j in range(A_w):
            while overlap_restricted_area(random_B_r[i][j], random_B_c[i][j], p_size, min_x, max_x, min_y, max_y):
                random_B_r[i][j] = np.random.randint(p, B_h - p)
                random_B_c[i][j] = np.random.randint(p, B_w - p)
    if create_f:
        f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])
    for i in range(A_h):
        for j in range(A_w):
            a = np.array([i, j])
            if create_f:
                b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
                f[i, j] = b
            else:
                b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
                if (i % 2 == 0) or (j % 2 == 0):
                    f[i, j] = b
                else:
                    b = f[i, j]
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)
    return f, dist, A_padding


def propagation(f, a, dist, A_padding, B, p_size, is_odd, min_x, max_x, min_y, max_y):
    A_h = np.size(A_padding, 0) - p_size + 1
    A_w = np.size(A_padding, 1) - p_size + 1
    # print(A_h, A_w)
    x = a[0]
    y = a[1]
    if is_odd:
        d_left = dist[max(x - 1, 0), y]
        d_up = dist[x, max(y - 1, 0)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_left, d_up]))
        if idx == 1 and (not overlap_restricted_area(f[max(x - 1, 0), y][0] + 1, f[max(x - 1, 0), y][1], p_size,
                                                     min_x, max_x, min_y, max_y)):
            f[x, y] = f[max(x - 1, 0), y]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)
        if idx == 2 and (not overlap_restricted_area(f[x, max(y - 1, 0)][0], f[x, max(y - 1, 0)][1] + 1, p_size,
                                                     min_x, max_x, min_y, max_y)):
            f[x, y] = f[x, max(y - 1, 0)]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)
    else:
        # print(dist.shape)
        # print(min(x + 1, A_h - 1), y)
        d_right = dist[min(x + 1, A_h - 1), y]
        d_down = dist[x, min(y + 1, A_w - 1)]
        d_current = dist[x, y]
        idx = np.argmin(np.array([d_current, d_right, d_down]))
        if idx == 1 and (
                not overlap_restricted_area(f[min(x + 1, A_h - 1), y][0] - 1, f[min(x + 1, A_h - 1), y][1], p_size,
                                            min_x, max_x, min_y, max_y)):
            f[x, y] = f[min(x + 1, A_h - 1), y]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)
        if idx == 2 and (
                not overlap_restricted_area(f[x, min(y + 1, A_w - 1)][0], f[x, min(y + 1, A_w - 1)][1] - 1, p_size,
                                            min_x, max_x, min_y, max_y)):
            f[x, y] = f[x, min(y + 1, A_w - 1)]
            dist[x, y] = cal_distance(a, f[x, y], A_padding, B, p_size)


def random_search(f, a, dist, A_padding, B, p_size, min_x, max_x, min_y, max_y, alpha=0.5):
    x = a[0]
    y = a[1]
    B_h = np.size(B, 0)
    B_w = np.size(B, 1)
    p = p_size // 2
    i = 4
    search_h = B_h * alpha ** i
    search_w = B_w * alpha ** i
    b_x = f[x, y][0]
    b_y = f[x, y][1]
    while search_h > 1 and search_w > 1:
        search_min_r = max(b_x - search_h, p)
        search_max_r = min(b_x + search_h, B_h - p)
        random_b_x = np.random.randint(search_min_r, search_max_r)
        search_min_c = max(b_y - search_w, p)
        search_max_c = min(b_y + search_w, B_w - p)
        random_b_y = np.random.randint(search_min_c, search_max_c)
        search_h = B_h * alpha ** i
        search_w = B_w * alpha ** i
        b = np.array([random_b_x, random_b_y])
        d = cal_distance(a, b, A_padding, B, p_size)
        if d < dist[x, y] and (not overlap_restricted_area(b[0], b[1], p_size, min_x, max_x, min_y, max_y)):
            dist[x, y] = d
            f[x, y] = b
        i += 1


def NNS(img, ref, p_size, itr, f, dist, img_padding, min_x, max_x, min_y, max_y):
    A_h = np.size(img, 0)
    A_w = np.size(img, 1)
    # print(A_h, A_w)
    # print(img_padding.shape)
    for itr in range(1, itr + 1):
        if itr % 2 == 0:
            for i in range(A_h - 1, -1, -1):
                for j in range(A_w - 1, -1, -1):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, False, min_x, max_x, min_y, max_y)
                    random_search(f, a, dist, img_padding, ref, p_size, min_x, max_x, min_y, max_y)
        else:
            for i in range(A_h):
                for j in range(A_w):
                    a = np.array([i, j])
                    propagation(f, a, dist, img_padding, ref, p_size, True, min_x, max_x, min_y, max_y)
                    random_search(f, a, dist, img_padding, ref, p_size, min_x, max_x, min_y, max_y)
        print("iteration: %d" % (itr))
    return f


def upsample_nnf(nnf):
    temp = np.zeros((nnf.shape[0], nnf.shape[1], 3))
    for x in range(nnf.shape[0]):
        for y in range(nnf.shape[1]):
            temp[x][y] = [nnf[x][y][0], nnf[x][y][1], 0]
    # img = np.zeros(shape=(size, size, 2), dtype=np.int)
    # small_size = nnf.shape[0]
    aw_ratio = 2  # ((size) // small_size)
    ah_ratio = 2  # ((size) // small_size)

    temp = cv2.resize(temp, None, fx=aw_ratio, fy=aw_ratio, interpolation=cv2.INTER_NEAREST)
    imge = np.zeros(shape=(temp.shape[0], temp.shape[1], 2), dtype=np.int)
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            pos = temp[i, j]
            imge[i, j] = pos[0] * aw_ratio, pos[1] * ah_ratio

    return imge


padding_size = [15, 15, 13, 9, 5, 2]
# padding_size = [9, 7, 5, 3, 3, 2]
iter_arr = [2, 2, 16, 40, 64, 64]


def main(img_path):
    # img_path = 'IMAGE/face.jpg'
    global img
    img = cv2.imread(img_path)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(min_x, min_y, height, width)
    global_min_x = min_x
    global_min_y = min_y
    global_max_x = max_x
    global_max_y = max_y
    # img = np.array(Image.open("./cup_a.jpg"))
    origin_ref = np.array(Image.open(img_path))
    # ref = cv2.pyrDown(origin_ref, (np.size(origin_ref, 0)//2, np.size(origin_ref, 1)//2))
    # Image.fromarray(ref).show()
    itr = 4
    start = time.time()
    # origin_img = origin_ref[min_x: max_x + 1, min_y:max_y + 1, :]
    # img = cv2.resize(origin_img, None, fx=2 ** (-4), fy=2 ** (-4), interpolation=cv2.INTER_NEAREST)
    f = 0
    depth = 3
    for l in range(depth, -1, -1):
        p_size = padding_size[l]
        gmin_x = global_min_x // (2 ** l)
        gmin_y = global_min_y // (2 ** l)
        gmax_x = global_max_x // (2 ** l)
        gmax_y = global_max_y // (2 ** l)
        # print(origin_ref.shape)
        # ref = cv2.resize(origin_ref, None, fx=2 ** (-l), fy=2 ** (-l), interpolation=cv2.INTER_LINEAR)
        ref = origin_ref
        for kk in range(l):
            ref = cv2.pyrDown(ref, (np.size(origin_ref, 0) // 2, np.size(origin_ref, 1) // 2))
        # print(ref.shape)
        # print(gmin_x, gmin_y, gmax_x, gmax_y)

        # !!!!!!!!!
        # img = ref[gmin_x: gmax_x + 1, gmin_y:gmax_y + 1, :]
        # !!!!!!!!!

        if l == depth:
            # img = ref[gmin_x: gmax_x + 1, gmin_y:gmax_y + 1, :]
            # img = np.zeros([gmax_x - gmin_x + 1, gmax_y - gmin_y + 1, 3])

            # !!!!!!!!!!
            img = np.random.randint(0, 256, size=(gmax_x - gmin_x + 1, gmax_y - gmin_y + 1, 3), dtype=np.uint8)
            # !!!!!!!!!!

            # print(np.shape(img)[0] // 4)
            f, dist, img_padding = initialization(img, ref, f, p_size, gmin_x, gmax_x, gmin_y, gmax_y, create_f=True)
        else:
            # print(img.shape)
            fake, dist, img_padding = initialization(img, ref, f, p_size, gmin_x, gmax_x, gmin_y, gmax_y,
                                                     create_f=False)
        # Image.fromarray(ref).show()
        # Image.fromarray(img).show()
        # print(img.shape)
        # print(img_padding.shape)
        for itter in range(iter_arr[l]):
            f = NNS(img, ref, p_size, itr, f, dist, img_padding, gmin_x, gmax_x, gmin_y, gmax_y)
            end = time.time()
            print(end - start)
            print(l, itter + 1, '/', iter_arr[l])
            tmp, img = reconstruction(f, img, ref, p_size, dist, gmin_x, gmax_x, gmin_y, gmax_y, itter)
            # if itter == iter_arr[l] - 1:
            #     Image.fromarray(tmp).show()
        # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # Image.fromarray(img).show()
        img = cv2.pyrUp(img, (np.size(img, 0) * 2, np.size(img, 1) * 2))
        f = upsample_nnf(f)
        # Image.fromarray(img).show()
    tmp = Image.fromarray(tmp)
    tmp.save("temp.jpg")
    return "temp.jpg"


if __name__ == '__main__':
    # img_path = 'IMAGE/face.jpg'
    img_path = 'IMAGE/birds.jpg'
    while True:
        img_path = main(img_path)

