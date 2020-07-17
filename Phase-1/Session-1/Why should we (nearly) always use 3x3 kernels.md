### 2) Why should we (nearly) always use 3x3 kernels?

While convoluting an image, we always nearly use **odd sized kernels or filters, because when we convolve on pixel level we will have symmetry around the center pixel** when we use odd sized kernels. But, if we use even sized kernels. We will lose that symmetry which can cause distortions, so mostly we don't use even sized kernels.

Now we have clear understanding about why we prefer odd sized kernels, but why 3X3 mainly. In odd sized kernels, we use 1X1 when we want to decrease the channels or any bottleneck situations. Other than that we use 3X3 mainly because compared to 5X5, 7X7 and etc... 3X3 have less parameters and most compilers are optimized for 3X3 kernels. Receptive Filed of 1 5X5 kernel is equal to 2 3X3 kernel, so in place of 5X5 we can do 2 3X3 operations and save some parameters too... as you can see 1 5X5 kernel have 25 parameters, but 2 3X3 have (9 * 2) = 18 parameters. Same applies for all higher odd number kernels, 1 7X7 equals to 3 3X3 kernels. 



### 3) How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

If the **input considered is 199X199**, we need to do **99 3X3 convolutions to get to 1X1**

199X199 > 197X197 > 195X195 > 193X193 > 191X191 > 189X189 > 187X187 > 185X185 > 183X183 > 181X181 > 179X179 > 177X177 > 175X175 > 173X173 > 171X171 > 169X169 > 167X167 > 165X165 > 163X163 > 161X161 > 159X159 > 157X157 > 155X155 > 153X153 > 151X151 > 149X149 > 147X147 > 145X145 > 143X143 > 141X141 > 139X139 > 137X137 > 135X135 > 133X133 > 131X131 > 129X129 > 127X127 > 125X125 > 123X123 > 121X121 > 119X119 > 117X117 > 115X115 > 113X113 > 111X111 > 109X109 > 107X107 > 105X105 > 103X103 > 101X101 > 99X99 > 97X97 > 95X95 > 93X93 > 91X91 > 89X89 > 87X87 > 85X85 > 83X83 > 81X81 > 79X79 > 77X77 > 75X75 > 73X73 > 71X71 > 69X69 > 67X67 > 65X65 > 63X63 > 61X61 > 59X59 > 57X57 > 55X55 > 53X53 > 51X51 > 49X49 > 47X47 > 45X45 > 43X43 > 41X41 > 39X39 > 37X37 > 35X35 > 33X33 > 31X31 > 29X29 > 27X27 > 25X25 > 23X23 > 21X21 > 19X19 > 17X17 > 15X15 > 13X13 > 11X11 > 9X9 > 7X7 > 5X5 > 3X3 > 1X1