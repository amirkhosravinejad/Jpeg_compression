"""import huffman
list = [8, 8, 34, 5, 10, 34, 6, 43, 127, 10, 10, 8, 10, 34, 10]
huff1 = huffman.huffman_main()
huff1.main(list)
#print(huffman.bit_num)
print()
list1 = [5, 6, 7]
huff1 = huffman.huffman_main()
huff1.main(list1)"""
#print(huffman.bit_num)
amplitude = bin(5).replace("0b", "")
amplitude_ = ""
for j in range(len(amplitude)):
                if amplitude[j] == '0':
                    amplitude_ += '1'
                else:
                    amplitude_ += '0'
print(amplitude_)
ampiltude = amplitude_