#import numpy as np
import math
import pickle
from PIL import Image as im
import numpy
import scipy
from scipy.fftpack import fft, dct
import huffman
import cv2

def YCbCr_withoutchroma_subsampling(img):

    for i in range(img.width):
        for j in range(img.height):
            data = img.getpixel((i,j))
        #print(data)
        #print(data) #(255, 255, 255)
            r = data[0]
            g = data[1]
            b = data[2]
            Y = 16 + (65.738 * r / 256) + (109.057 * g / 256) + (25.064 * b / 256)
            Cb = 128 - (37.945 * r / 256) - (74.494 * g / 256) + (112.439 * b / 256)
            Cr = 128 + (112.439 * r / 256) - (95.154 * g / 256) - (18.285 * b / 256)
            # Y = 16 + (65.738 * r / 256) + (109.057 * g / 256) + (25.064 * b / 256)
            #Cb = 128 - (37.945 * r / 256) - (74.494 * g / 256) + (112.439 * b / 256)
            # Cr = 128 + (112.439 * r / 256) - (95.154 * g / 256) - (18.285 * b / 256)
            img.putpixel((i,j), (int(Y), int(Cb), int(Cr)))
    img.save('wochroma_subsample.png')

def C(u):
    if u == 0:
        return (math.sqrt(2)) / 2
    return 1

def dct_2D(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )
    #return cv2.dct(a)
    """F = [[0] * 8] * 8
    for u in range(8):
        for v in range(8):
            c = C(u) * C(v) / 4
            sum = 0
            for i in range(8):
                for j in range(8):
                    sum += math.cos((2 * i + 1) * math.pi * u / 16) * math.cos((2 * j + 1) * math.pi * v / 16) * f[i][j][index]
            F[u][v] = c * sum * 10 ** 13        
    return F"""
            

def block_8x8(width, height):
    dcts = [[0.0] * height] * width
    for i in range(0, width, 8):
        for j in range(0, height, 8):
            arr = [[0] * 8 ] * 8
            for b_x in range(i, i + 8):
                for b_y in range(j, j + 8):
                    if (b_x >= width) or (b_y >= height):
                        data = (16, 128, 128)
                    else:    
                        data = image_.getpixel((b_x, b_y))
                    arr[b_x - i][b_y - j] = data
            #if i == 0 and j == 0:        
                #print(arr)
            #dct_out_Y = dct_2D(arr, 0)
            #dct_out_Cb = dct_2D(arr, 1)
            #dct_out_Cr = dct_2D(arr, 2)
            #print(dct_out_Y)
            """list_asli = []
            for s1 in range(8):
                lis_ = []
                for s2 in range(8):
                    lis_.append((dct_out_Y[s1][s2], dct_out_Cb[s1][s2], dct_out_Cr[s1][s2]))
                list_asli.append(lis_)"""
            dcts[i][j] = dct_2D(arr)
            #print("i = {}, j = {}".format(i, j))       
            #print(dcts[i][j]) 
            #if i == 400 and j == 80:       
                #print(i, j, dcts[i][j][0][0][0])
                #for k in range(8):
                    #for l in range(8):
                        #print(i, j, dcts[i][j][k][l][0])               
            #for item in arr:
                #print(item)
    return dcts            

def quantization_and_round(dcts, height, width):
    F_ = [[0] * height] * width
    # luminance quantization table
    luminance_quant_table = numpy.array(
    [ 16,  11,  10,  16,  24,  40,  51,  61,
      12,  12,  14,  19,  26,  58,  60,  55,
      14,  13,  16,  24,  40,  57,  69,  56,
      14,  17,  22,  29,  51,  87,  80,  62,
      18,  22,  37,  56,  68, 109, 103,  77,
      24,  35,  55,  64,  81, 104, 113,  92,
      49,  64,  78,  87, 103, 121, 120, 101,
      72,  92,  95,  98, 112, 100, 103,  99],dtype=int)
    luminance_quant_table = luminance_quant_table.reshape([8,8])
    # chrominance quantization table
    chrominance_quant_table = numpy.array(
    [ 17,  18,  24,  47,  99,  99,  99,  99,
      18,  21,  26,  66,  99,  99,  99,  99,
      24,  26,  56,  99,  99,  99,  99,  99,
      47,  66,  99,  99,  99,  99,  99,  99,
      99,  99,  99,  99,  99,  99,  99,  99,
      99,  99,  99,  99,  99,  99,  99,  99,
      99,  99,  99,  99,  99,  99,  99,  99,
      99,  99,  99,  99,  99,  99,  99,  99],dtype=int)
    chrominance_quant_table = chrominance_quant_table.reshape([8,8])
    list_of_firsts = []
    for i in range(0, width, 8):
        for j in range(0, height, 8):
            #if i == 0 and j == 0 or i == 0 and j == 8:
                #print(i, j)
                
                list_kl = []
                for k in range(8):
                    list_l = []
                    for l in range(8):
                        # in each block we extract Y, Cb and Cr and then
                        # we round them by dividing Y value to the corresponding
                        # value in luminance quantization table and doing the same
                        # with dividing Cb, Cr value to the related index in-
                        # chrominance quantization table. after that we round them
                        Y = dcts[i][j][k][l][0]
                        Ybar = round(Y / luminance_quant_table[k][l])
                        
                        Cb = dcts[i][j][k][l][1]
                        Cbbar = round(Cb / chrominance_quant_table[k][l])

                        Cr = dcts[i][j][k][l][2]
                        Crbar = round(Cr / chrominance_quant_table[k][l])
                        list_l.append((Ybar, Cbbar, Crbar))
                        #print("Y: ", Y, "Cb: ", Cb, "Cr: ", Cr)
                        #print(luminance_quant_table[k][l])
                        #print(chrominance_quant_table[k][l])
                        #print(k, l)
                        #print(arr[k][l])
                    #print("list_l:", list_l)    
                    list_kl.append(list_l)    
                #print('oken')        
                #for item in list_kl:
                    #print(item)
                
                   #print(list_kl)
                zigzag(list_kl)
                #if i != 0 and j != 0: 
                    #prev_pcm = list_kl[0][0] 
                #list_kl[0][0] = (list_kl[0][0][0] - prev_pcm[0], list_kl[0][0][1] - prev_pcm[1], list_kl[0][0][2] - prev_pcm[2])
                #print(i, j, prev_pcm)
                #print("avali: ", list_kl[0][0])
                list_of_firsts.append(list_kl[0][0])
                #print("ta hala", list_of_firsts)
                F_[i][j] = list_kl
                #print(F_[i][j]) 
    dpcm(list_of_firsts)      
    return F_

def dpcm(list_of_firsts):
    #print("sal")
    #print(len(list_of_firsts))
    new_ = [(list_of_firsts[0][0], list_of_firsts[0][1], list_of_firsts[0][2])]
    for i in range(1, len(list_of_firsts)):
        new_.append((list_of_firsts[i][0] - list_of_firsts[i - 1][0], list_of_firsts[i][1] - list_of_firsts[i - 1][1], list_of_firsts[i][2] - list_of_firsts[i - 1][2]))
    list_of_size = []
    list_of_sizeCb = []
    list_of_sizeCr = []
    #print(new_)
    for i in range(0, len(new_)):
        amplitude = bin(new_[i][0]).replace("0b", "")
        amplitudeCb = bin(new_[i][1]).replace("0b", "")
        amplitudeCr = bin(new_[i][2]).replace("0b", "")
        if (new_[i][0] < 0):
            amplitude_ = ""
            for j in range(len(amplitude)):
                if amplitude[j] == '0':
                    amplitude_ += '1'
                else:
                    amplitude_ += '0'
            amplitude = amplitude_
        if (new_[i][1] < 0):
            amplitude_ = ""
            for j in range(len(amplitudeCb)):
                if amplitudeCb[j] == '0':
                    amplitude_ += '1'
                else:
                    amplitude_ += '0'
            amplitudeCb = amplitude_
        if (new_[i][2] < 0):
            amplitude_ = ""
            for j in range(len(amplitudeCr)):
                if amplitudeCr[j] == '0':
                    amplitude_ += '1'
                else:
                    amplitude_ += '0'
            amplitude = amplitude_        
        size = len(amplitude)
        size_cb = len(amplitudeCb)
        size_cr = len(amplitudeCr)
        list_of_size.append(size)
        list_of_sizeCb.append(size_cb)
        list_of_sizeCr.append(size_cr) 
    dpcm = {}      
    file_dpcm = open("f_dcpm", "ab")   
    h1 = huffman.huffman_main()
    dpcm["Y"] = h1.main(list_of_size)
    
    h1 = huffman.huffman_main()
    dpcm["Cb"] = h1.main(list_of_sizeCb)

    h1 = huffman.huffman_main()
    dpcm["Cr"] = h1.main(list_of_sizeCr)

    pickle.dump(dpcm, file_dpcm)
    #for i in range(0, len(list_of_firsts)):
        #print("new:", new_[i])
        #print("dpcm:", list_of_firsts)  

def zigzag(quantized_block):
    vector_64 = []
    vector_64.append(quantized_block[0][0])
    #print(vector_64)
    for i in range(1, 8):
        if i % 2 == 1:
            for j in range(i + 1):
                vector_64.append(quantized_block[j][i - j])
                #print(j, i - j)
        else:
            for j in range(i, -1, -1):
                vector_64.append(quantized_block[j][i - j])
                #print(j, i - j)
    for i in range(1, 8):
        sum = 7 + i
        if i % 2 == 1:
            for j in range(7, i - 1, -1):
                vector_64.append(quantized_block[j][sum - j])
                #print(j, sum - j)
        else:
            for j in range(i, 8):
                vector_64.append(quantized_block[j][sum - j])
                #print(j, sum - j)
    AC_components = RLC(vector_64)
    #return AC_components

def RLC(list):
    pre_freq_zy = pre_freq_zCb = 0
    pre_freq_zCr = 0

    rlc_list_Y = []
    rlc_list_Cb = []
    rlc_list_Cr = []
    for i in range(1, len(list)):
        #if i == 0:
            #print(list[i])
        if list[i][0] != 0:
            #rlc_list_Y.append((pre_freq_zy, list[i][0]))
            rlc_list_Y.append(pre_freq_zy)
            pre_freq_zy = 0
        else:
            pre_freq_zy += 1
        if list[i][1] != 0:
            #rlc_list_Cb.append((pre_freq_zCb, list[i][1]))
            rlc_list_Cb.append(pre_freq_zCb)
            pre_freq_zCb = 0
        else:
            pre_freq_zCb += 1
        if list[i][2] != 0:
            #rlc_list_Cr.append((pre_freq_zCr, list[i][2]))
            rlc_list_Cr.append(pre_freq_zCr)
            pre_freq_zCr = 0
        else:
            pre_freq_zCr += 1        
    rlc_list_Y.append(0)
    rlc_list_Cb.append(0)
    rlc_list_Cr.append(0)
    #print(rlc_list_Y)
    AC = {}      
    AC_f = open("AC_file", "ab")   
    h1 = huffman.huffman_main()
    AC["Y"] = h1.main(rlc_list_Y)
    
    h1 = huffman.huffman_main()
    AC["Cb"] = h1.main(rlc_list_Cb)

    h1 = huffman.huffman_main()
    AC["Cr"] = h1.main(rlc_list_Cr)

    pickle.dump(AC, AC_f)
    #return rlc_list         


image_ = im.open('photo1.png')
#image_.show()
width = image_.size[0] 
height = image_.size[1]
#print(width, height)
#YCbCr_withoutchroma_subsampling(image_)
for i in range(width):
    for j in range(height):
            data = image_.getpixel((i,j))
        #print(data)
        #print(data) #(255, 255, 255)
            r = data[0]
            g = data[1]
            b = data[2]
            Y = 16 + (65.738 * r / 256) + (109.057 * g / 256) + (25.064 * b / 256)
            if (i % 2 == 0 and j % 2 == 0):
                Cb = 128 - (37.945 * r / 256) - (74.494 * g / 256) + (112.439 * b / 256)
                Cr = 128 + (112.439 * r / 256) - (95.154 * g / 256) - (18.285 * b / 256)
            else:
                Cb = image_.getpixel((i - (i % 2), j - (j % 2)))[1]
                Cr = image_.getpixel((i - (i % 2), j - (j % 2)))[2]
            image_.putpixel((i,j), (int(Y), int(Cb), int(Cr)))
#image_.show()            
#image_.save('with_sub.png')
dcts = block_8x8(width, height)
blocks = quantization_and_round(dcts, height, width)
#use_hauffman_code(blocks, width, height)
#r = np.asarray(image_)
#for i in r:
    #print(i)
    