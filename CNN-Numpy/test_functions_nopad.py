#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
from functions import conv2d_forward, conv2d_backward, avgpool2d_forward, avgpool2d_backward
import numpy as np


def get_fake_data():
    inp = np.array([[[[2, 8, 9, 8, 0, 0],
                     [2, 4, 5, 2, 5, 3],
                     [8, 3, 4, 0, 0, 2],
                     [3, 8, 4, 9, 4, 0],
                     [9, 2, 2, 9, 5, 8],
                     [5, 9, 5, 9, 9, 0]],

                    [[8, 8, 4, 3, 5, 4],
                     [7, 0, 1, 2, 6, 7],
                     [3, 1, 5, 7, 5, 3],
                     [3, 3, 9, 4, 1, 2],
                     [1, 5, 3, 0, 9, 3],
                     [8, 6, 1, 9, 8, 6]],

                    [[1, 4, 1, 1, 4, 5],
                     [1, 5, 8, 6, 1, 1],
                     [7, 6, 9, 0, 2, 2],
                     [1, 4, 4, 5, 0, 8],
                     [8, 7, 3, 3, 9, 8],
                     [2, 5, 6, 5, 3, 9]]],


                   [[[7, 0, 2, 7, 4, 7],
                     [9, 3, 0, 6, 3, 9],
                     [7, 6, 2, 4, 6, 8],
                     [6, 6, 3, 7, 2, 0],
                     [7, 9, 6, 7, 0, 8],
                     [2, 0, 1, 8, 4, 4]],

                    [[6, 2, 5, 3, 0, 5],
                     [3, 8, 9, 6, 0, 5],
                     [6, 2, 3, 8, 7, 1],
                     [4, 0, 3, 8, 8, 0],
                     [1, 4, 7, 1, 0, 0],
                     [0, 6, 9, 7, 2, 4]],

                    [[7, 2, 2, 3, 3, 7],
                     [8, 2, 1, 0, 7, 5],
                     [2, 7, 9, 9, 0, 0],
                     [8, 1, 2, 6, 5, 6],
                     [1, 4, 6, 3, 3, 4],
                     [5, 7, 2, 2, 5, 0]]],


                   [[[7, 6, 6, 5, 1, 7],
                     [6, 5, 8, 4, 1, 8],
                     [3, 6, 2, 7, 5, 4],
                     [1, 9, 7, 0, 8, 3],
                     [5, 5, 8, 0, 6, 4],
                     [6, 9, 2, 6, 5, 4]],

                    [[5, 3, 1, 9, 8, 4],
                     [0, 3, 9, 7, 5, 8],
                     [2, 9, 5, 8, 4, 4],
                     [5, 7, 9, 8, 9, 0],
                     [7, 5, 5, 7, 9, 4],
                     [7, 3, 5, 3, 0, 4]],

                    [[7, 0, 9, 7, 1, 8],
                     [2, 9, 0, 0, 1, 3],
                     [0, 2, 9, 7, 2, 2],
                     [9, 0, 7, 3, 8, 2],
                     [0, 1, 4, 5, 1, 3],
                     [9, 0, 0, 6, 0, 5]]],


                   [[[0, 7, 1, 4, 1, 1],
                     [7, 5, 9, 2, 8, 6],
                     [8, 1, 0, 9, 2, 1],
                     [1, 1, 8, 7, 1, 2],
                     [8, 7, 5, 3, 8, 6],
                     [3, 7, 5, 1, 0, 5]],

                    [[3, 2, 3, 6, 5, 6],
                     [2, 5, 4, 0, 7, 4],
                     [4, 0, 6, 6, 8, 7],
                     [9, 6, 3, 6, 9, 9],
                     [9, 7, 8, 4, 7, 4],
                     [7, 2, 5, 1, 8, 4]],

                    [[5, 6, 9, 3, 7, 6],
                     [1, 2, 8, 1, 8, 6],
                     [1, 9, 6, 3, 8, 9],
                     [5, 9, 6, 2, 0, 2],
                     [7, 2, 3, 7, 6, 0],
                     [4, 3, 2, 7, 3, 3]]]])

    w = np.array([[[[1, 8],
                 [2, 3]],

                [[6, 5],
                 [5, 0]],

                [[0, 5],
                 [4, 2]]],


               [[[4, 8],
                 [6, 2]],

                [[2, 9],
                 [8, 1]],

                [[6, 1],
                 [6, 8]]],


               [[[7, 2],
                 [6, 7]],

                [[8, 1],
                 [6, 1]],

                [[9, 7],
                 [8, 0]]],


               [[[8, 0],
                 [9, 5]],

                [[2, 2],
                 [4, 2]],

                [[6, 5],
                 [4, 0]]]])

    b = np.array([9, 6, 3, 6])

    grad_out = np.array([[[[3, 9, 9, 1, 0],
                         [4, 8, 1, 7, 2],
                         [5, 0, 0, 1, 4],
                         [3, 9, 8, 1, 5],
                         [0, 1, 2, 9, 7]],

                        [[1, 4, 0, 7, 4],
                         [5, 5, 2, 9, 4],
                         [6, 2, 0, 9, 2],
                         [2, 1, 7, 1, 9],
                         [4, 8, 8, 4, 9]],

                        [[4, 4, 9, 7, 2],
                         [2, 7, 3, 6, 7],
                         [5, 6, 6, 0, 3],
                         [2, 2, 2, 2, 9],
                         [6, 0, 2, 3, 6]],

                        [[3, 6, 7, 2, 8],
                         [3, 0, 8, 1, 0],
                         [8, 9, 7, 2, 6],
                         [6, 1, 6, 0, 1],
                         [2, 4, 8, 2, 5]]],


                       [[[2, 4, 8, 8, 5],
                         [7, 0, 5, 4, 3],
                         [8, 8, 0, 2, 4],
                         [7, 4, 1, 0, 6],
                         [0, 6, 1, 0, 2]],

                        [[9, 1, 5, 6, 5],
                         [6, 5, 8, 6, 9],
                         [4, 0, 8, 7, 1],
                         [3, 1, 3, 2, 7],
                         [2, 9, 3, 2, 0]],

                        [[1, 4, 4, 7, 0],
                         [3, 8, 7, 7, 2],
                         [9, 7, 5, 1, 0],
                         [9, 0, 9, 2, 9],
                         [5, 8, 7, 3, 2]],

                        [[7, 0, 5, 1, 6],
                         [0, 8, 1, 2, 5],
                         [6, 6, 3, 0, 7],
                         [4, 2, 8, 4, 7],
                         [4, 8, 6, 8, 0]]],


                       [[[9, 5, 8, 3, 8],
                         [7, 5, 8, 5, 7],
                         [3, 5, 4, 8, 5],
                         [7, 4, 5, 5, 0],
                         [0, 9, 8, 5, 0]],

                        [[2, 2, 2, 3, 6],
                         [5, 7, 1, 8, 1],
                         [9, 6, 7, 5, 6],
                         [1, 2, 5, 8, 2],
                         [2, 3, 5, 1, 9]],

                        [[1, 4, 9, 7, 2],
                         [6, 2, 0, 8, 8],
                         [1, 2, 8, 3, 6],
                         [6, 9, 1, 6, 5],
                         [8, 6, 6, 7, 4]],

                        [[0, 5, 7, 0, 0],
                         [8, 6, 3, 8, 4],
                         [4, 5, 6, 4, 9],
                         [5, 8, 5, 8, 6],
                         [9, 0, 4, 7, 9]]],


                       [[[2, 1, 3, 5, 9],
                         [2, 8, 2, 3, 7],
                         [6, 8, 0, 0, 1],
                         [7, 1, 2, 2, 2],
                         [3, 0, 4, 4, 1]],

                        [[3, 3, 2, 9, 7],
                         [3, 2, 9, 4, 1],
                         [4, 4, 3, 0, 5],
                         [9, 4, 1, 4, 8],
                         [5, 5, 2, 1, 9]],

                        [[4, 6, 1, 5, 8],
                         [4, 3, 0, 7, 9],
                         [8, 0, 3, 3, 1],
                         [0, 3, 1, 3, 2],
                         [2, 3, 5, 4, 1]],

                        [[8, 6, 6, 3, 4],
                         [1, 4, 3, 0, 5],
                         [5, 9, 0, 8, 8],
                         [4, 3, 8, 3, 4],
                         [0, 9, 9, 3, 4]]]])

    return inp, w, b, grad_out


def get_conv_answer():
    out = np.array([[[[ 248.,  221.,  191.,  135.,  139.],
                     [ 190.,  163.,  145.,  137.,  157.],
                     [ 151.,  187.,  184.,  156.,  109.],
                     [ 204.,  201.,  248.,  142.,  200.],
                     [ 195.,  182.,  192.,  264.,  275.]],

                    [[ 298.,  316.,  288.,  187.,  186.],
                     [ 240.,  256.,  241.,  226.,  207.],
                     [ 224.,  287.,  331.,  192.,  171.],
                     [ 300.,  310.,  273.,  284.,  289.],
                     [ 330.,  303.,  254.,  396.,  411.]],

                    [[ 232.,  288.,  249.,  238.,  220.],
                     [ 269.,  248.,  299.,  157.,  178.],
                     [ 298.,  297.,  336.,  225.,  114.],
                     [ 247.,  279.,  313.,  274.,  312.],
                     [ 367.,  314.,  263.,  394.,  398.]],

                    [[ 150.,  206.,  198.,  199.,  175.],
                     [ 196.,  195.,  236.,  117.,  127.],
                     [ 239.,  261.,  257.,  179.,   88.],
                     [ 205.,  220.,  200.,  254.,  247.],
                     [ 315.,  247.,  197.,  357.,  301.]]],


                   [[[ 150.,  128.,  194.,  146.,  200.],
                     [ 194.,  184.,  226.,  212.,  205.],
                     [ 229.,  132.,  208.,  238.,  202.],
                     [ 150.,  148.,  254.,  182.,  133.],
                     [ 172.,  220.,  216.,  118.,  143.]],

                    [[ 264.,  196.,  218.,  239.,  271.],
                     [ 366.,  301.,  310.,  234.,  291.],
                     [ 267.,  195.,  311.,  397.,  271.],
                     [ 245.,  266.,  337.,  271.,  132.],
                     [ 258.,  308.,  277.,  223.,  174.]],

                    [[ 344.,  151.,  223.,  225.,  268.],
                     [ 328.,  243.,  240.,  341.,  280.],
                     [ 347.,  262.,  310.,  355.,  218.],
                     [ 291.,  236.,  306.,  289.,  248.],
                     [ 177.,  303.,  330.,  244.,  182.]],

                    [[ 286.,  127.,  147.,  194.,  211.],
                     [ 287.,  187.,  144.,  249.,  231.],
                     [ 257.,  230.,  241.,  267.,  140.],
                     [ 239.,  233.,  237.,  234.,  150.],
                     [ 148.,  229.,  228.,  229.,  136.]]],


                   [[[ 162.,  216.,  214.,  169.,  235.],
                     [ 153.,  230.,  238.,  194.,  213.],
                     [ 217.,  243.,  265.,  226.,  206.],
                     [ 209.,  267.,  192.,  281.,  184.],
                     [ 236.,  192.,  166.,  215.,  168.]],

                    [[ 294.,  235.,  349.,  262.,  232.],
                     [ 189.,  432.,  335.,  268.,  267.],
                     [ 278.,  319.,  401.,  341.,  306.],
                     [ 318.,  329.,  328.,  308.,  280.],
                     [ 293.,  242.,  255.,  286.,  207.]],

                    [[ 260.,  330.,  339.,  268.,  265.],
                     [ 220.,  293.,  317.,  286.,  206.],
                     [ 253.,  361.,  369.,  344.,  301.],
                     [ 268.,  323.,  333.,  308.,  353.],
                     [ 332.,  227.,  264.,  269.,  221.]],

                    [[ 213.,  258.,  305.,  206.,  173.],
                     [ 200.,  242.,  227.,  223.,  158.],
                     [ 186.,  301.,  280.,  240.,  239.],
                     [ 200.,  264.,  275.,  194.,  268.],
                     [ 244.,  205.,  217.,  188.,  174.]]],


                   [[[ 170.,  182.,  183.,  165.,  221.],
                     [ 164.,  226.,  150.,  232.,  254.],
                     [ 182.,  174.,  242.,  205.,  204.],
                     [ 261.,  233.,  216.,  195.,  228.],
                     [ 256.,  208.,  201.,  206.,  200.]],

                    [[ 217.,  286.,  305.,  217.,  346.],
                     [ 291.,  278.,  247.,  351.,  395.],
                     [ 257.,  299.,  327.,  281.,  265.],
                     [ 328.,  325.,  370.,  310.,  283.],
                     [ 357.,  282.,  268.,  289.,  292.]],

                    [[ 232.,  333.,  306.,  245.,  363.],
                     [ 193.,  258.,  334.,  241.,  383.],
                     [ 278.,  312.,  319.,  319.,  308.],
                     [ 412.,  343.,  298.,  290.,  293.],
                     [ 372.,  283.,  272.,  260.,  296.]],

                    [[ 186.,  279.,  240.,  189.,  278.],
                     [ 189.,  173.,  244.,  225.,  271.],
                     [ 211.,  225.,  236.,  282.,  218.],
                     [ 304.,  256.,  246.,  229.,  222.],
                     [ 264.,  237.,  203.,  181.,  205.]]]])

    grad_inp = np.array([[[[  59.,  141.,  240.,  184.,  172.,   36.],
                         [ 125.,  327.,  440.,  352.,  391.,  124.],
                         [ 205.,  359.,  313.,  246.,  317.,  117.],
                         [ 221.,  310.,  372.,  294.,  255.,  197.],
                         [ 158.,  211.,  356.,  248.,  381.,  241.],
                         [  78.,  146.,  175.,  154.,  215.,  106.]],

                        [[  58.,  140.,  237.,  148.,  119.,   54.],
                         [ 115.,  323.,  289.,  286.,  294.,   75.],
                         [ 182.,  305.,  187.,  216.,  255.,   64.],
                         [ 185.,  236.,  238.,  246.,  229.,  134.],
                         [ 127.,  161.,  282.,  236.,  364.,  152.],
                         [  76.,   99.,  134.,  129.,  174.,   25.]],

                        [[  60.,  155.,  230.,  260.,  161.,   58.],
                         [ 128.,  277.,  367.,  310.,  308.,   95.],
                         [ 203.,  392.,  313.,  295.,  273.,  109.],
                         [ 194.,  245.,  253.,  181.,  311.,  126.],
                         [ 154.,  212.,  287.,  233.,  378.,  193.],
                         [  80.,  100.,  170.,  160.,  200.,   86.]]],


                       [[[ 101.,  126.,  144.,  201.,  199.,   80.],
                         [ 179.,  354.,  307.,  401.,  371.,  155.],
                         [ 203.,  423.,  384.,  300.,  338.,  106.],
                         [ 262.,  359.,  388.,  222.,  277.,  171.],
                         [ 197.,  330.,  416.,  311.,  247.,  150.],
                         [  78.,  245.,  248.,  190.,   81.,   20.]],

                        [[  52.,  164.,  133.,  217.,  155.,   82.],
                         [ 194.,  258.,  302.,  355.,  251.,  125.],
                         [ 241.,  342.,  285.,  252.,  250.,   64.],
                         [ 278.,  240.,  258.,  176.,  228.,  131.],
                         [ 181.,  221.,  335.,  167.,  245.,   42.],
                         [  62.,  197.,  128.,   88.,   43.,    2.]],

                        [[ 105.,  103.,  145.,  203.,  166.,   60.],
                         [ 161.,  342.,  348.,  382.,  325.,  113.],
                         [ 229.,  424.,  398.,  299.,  230.,  134.],
                         [ 275.,  299.,  294.,  287.,  311.,  151.],
                         [ 215.,  299.,  394.,  244.,  263.,   92.],
                         [  68.,  190.,  186.,   94.,   40.,    4.]]],


                       [[[  24.,  171.,  199.,  162.,  108.,  116.],
                         [ 169.,  332.,  353.,  421.,  347.,  130.],
                         [ 230.,  412.,  350.,  404.,  484.,  199.],
                         [ 195.,  372.,  356.,  437.,  445.,  140.],
                         [ 237.,  331.,  420.,  453.,  407.,  149.],
                         [ 141.,  177.,  193.,  217.,  260.,   91.]],

                        [[  66.,  140.,  195.,  161.,  125.,   96.],
                         [ 183.,  262.,  312.,  287.,  347.,   68.],
                         [ 195.,  317.,  286.,  369.,  343.,  120.],
                         [ 211.,  299.,  287.,  332.,  336.,   65.],
                         [ 185.,  291.,  310.,  374.,  223.,  122.],
                         [ 100.,  133.,  141.,  122.,  154.,   31.]],

                        [[  21.,  132.,  215.,  221.,  121.,   60.],
                         [ 188.,  336.,  270.,  342.,  345.,  176.],
                         [ 225.,  291.,  336.,  382.,  418.,  140.],
                         [ 180.,  414.,  398.,  378.,  404.,  125.],
                         [ 240.,  329.,  300.,  416.,  404.,   98.],
                         [ 112.,  118.,  168.,  166.,  140.,   72.]]],


                       [[[ 106.,  151.,  110.,  142.,  247.,  144.],
                         [ 168.,  307.,  307.,  327.,  411.,  199.],
                         [ 173.,  319.,  283.,  226.,  283.,  161.],
                         [ 204.,  418.,  236.,  206.,  305.,  144.],
                         [ 141.,  311.,  300.,  246.,  274.,  138.],
                         [  48.,  162.,  207.,  161.,  155.,   48.]],

                        [[  66.,  129.,   92.,  140.,  257.,  124.],
                         [ 142.,  239.,  187.,  333.,  369.,   86.],
                         [ 180.,  257.,  231.,  174.,  187.,   87.],
                         [ 198.,  298.,  152.,  142.,  209.,  114.],
                         [ 167.,  198.,  221.,  215.,  194.,  113.],
                         [  67.,  101.,  128.,   89.,  110.,   18.]],

                        [[ 102.,  189.,  137.,  171.,  222.,  128.],
                         [ 150.,  231.,  237.,  269.,  425.,  198.],
                         [ 188.,  305.,  240.,  267.,  312.,   79.],
                         [ 210.,  269.,  198.,  207.,  214.,   94.],
                         [ 146.,  295.,  270.,  242.,  275.,  109.],
                         [  58.,  136.,  144.,   90.,   98.,   74.]]]])

    grad_w = np.array([[[[ 1991.,  2011.],
                         [ 1825.,  1906.]],

                        [[ 1966.,  1940.],
                         [ 1999.,  1925.]],

                        [[ 1692.,  1858.],
                         [ 1726.,  1805.]]],


                       [[[ 1944.,  1843.],
                         [ 2034.,  2213.]],

                        [[ 2270.,  2102.],
                         [ 2290.,  2208.]],

                        [[ 1865.,  1922.],
                         [ 1877.,  1798.]]],


                       [[[ 2092.,  1970.],
                         [ 2019.,  1968.]],

                        [[ 2074.,  2076.],
                         [ 2188.,  2053.]],

                        [[ 1741.,  2018.],
                         [ 1761.,  1781.]]],


                       [[[ 2168.,  2010.],
                         [ 2222.,  2126.]],

                        [[ 2282.,  2178.],
                         [ 2286.,  2160.]],

                        [[ 1961.,  2094.],
                         [ 1949.,  1768.]]]])

    grad_b = np.array([410, 440, 435, 460])

    return out, grad_inp, grad_w, grad_b

def get_pool_data_answer():
    pool_inp = np.array([[[[8, 2, 2, 3, 0, 3],
                         [9, 9, 4, 0, 6, 8],
                         [9, 6, 6, 4, 1, 4],
                         [0, 6, 5, 4, 9, 9],
                         [4, 8, 9, 0, 5, 3],
                         [5, 2, 6, 1, 4, 4]],

                        [[3, 1, 7, 8, 0, 2],
                         [2, 4, 9, 2, 8, 2],
                         [6, 0, 9, 1, 9, 5],
                         [9, 1, 1, 2, 1, 0],
                         [4, 0, 3, 2, 3, 0],
                         [3, 8, 1, 1, 2, 5]],

                        [[7, 9, 4, 7, 1, 2],
                         [2, 0, 8, 6, 2, 9],
                         [1, 7, 7, 8, 1, 6],
                         [8, 9, 5, 2, 9, 7],
                         [9, 1, 9, 9, 5, 0],
                         [3, 3, 0, 7, 0, 8]]],


                       [[[2, 6, 3, 3, 7, 7],
                         [7, 9, 5, 3, 0, 0],
                         [6, 2, 7, 1, 2, 0],
                         [5, 0, 8, 9, 2, 4],
                         [8, 1, 7, 5, 8, 0],
                         [6, 4, 6, 8, 6, 9]],

                        [[4, 8, 1, 9, 9, 2],
                         [5, 0, 7, 9, 8, 3],
                         [4, 9, 3, 8, 1, 0],
                         [8, 6, 3, 3, 6, 4],
                         [1, 4, 1, 2, 3, 9],
                         [8, 3, 6, 5, 4, 4]],

                        [[4, 5, 1, 5, 1, 0],
                         [9, 0, 5, 1, 8, 4],
                         [5, 5, 3, 5, 4, 7],
                         [9, 3, 3, 5, 0, 5],
                         [3, 1, 9, 2, 3, 7],
                         [1, 6, 7, 9, 9, 8]]],


                       [[[4, 8, 6, 9, 3, 3],
                         [4, 0, 8, 6, 1, 6],
                         [5, 8, 5, 6, 2, 4],
                         [2, 8, 8, 4, 2, 9],
                         [7, 5, 3, 6, 9, 3],
                         [0, 1, 5, 6, 9, 9]],

                        [[2, 4, 5, 7, 6, 4],
                         [0, 5, 3, 1, 1, 0],
                         [6, 9, 4, 7, 6, 5],
                         [4, 6, 6, 0, 2, 4],
                         [4, 0, 1, 2, 0, 0],
                         [7, 2, 8, 9, 0, 7]],

                        [[7, 4, 0, 0, 4, 1],
                         [2, 6, 7, 0, 5, 2],
                         [3, 2, 7, 8, 7, 9],
                         [2, 5, 6, 4, 4, 8],
                         [5, 3, 1, 7, 9, 2],
                         [2, 8, 5, 9, 1, 1]]],


                       [[[0, 5, 4, 0, 3, 1],
                         [7, 7, 2, 8, 1, 6],
                         [8, 9, 5, 5, 7, 3],
                         [9, 9, 1, 2, 6, 6],
                         [7, 8, 4, 7, 5, 6],
                         [2, 2, 1, 4, 0, 0]],

                        [[5, 4, 2, 0, 5, 7],
                         [9, 3, 2, 0, 7, 3],
                         [7, 7, 7, 2, 1, 7],
                         [4, 4, 0, 4, 0, 0],
                         [5, 0, 6, 1, 3, 7],
                         [4, 2, 2, 2, 9, 9]],

                        [[2, 3, 8, 4, 4, 0],
                         [5, 9, 3, 7, 2, 5],
                         [2, 9, 4, 6, 4, 5],
                         [7, 9, 0, 9, 6, 1],
                         [5, 1, 5, 7, 7, 9],
                         [8, 7, 7, 2, 2, 5]]]])

    pool_out = np.array([[[[ 7.  ,  2.25,  4.25],
                         [ 5.25,  4.75,  5.75],
                         [ 4.75,  4.  ,  4.  ]],

                        [[ 2.5 ,  6.5 ,  3.  ],
                         [ 4.  ,  3.25,  3.75],
                         [ 3.75,  1.75,  2.5 ]],

                        [[ 4.5 ,  6.25,  3.5 ],
                         [ 6.25,  5.5 ,  5.75],
                         [ 4.  ,  6.25,  3.25]]],


                       [[[ 6.  ,  3.5 ,  3.5 ],
                         [ 3.25,  6.25,  2.  ],
                         [ 4.75,  6.5 ,  5.75]],

                        [[ 4.25,  6.5 ,  5.5 ],
                         [ 6.75,  4.25,  2.75],
                         [ 4.  ,  3.5 ,  5.  ]],

                        [[ 4.5 ,  3.  ,  3.25],
                         [ 5.5 ,  4.  ,  4.  ],
                         [ 2.75,  6.75,  6.75]]],


                       [[[ 4.  ,  7.25,  3.25],
                         [ 5.75,  5.75,  4.25],
                         [ 3.25,  5.  ,  7.5 ]],

                        [[ 2.75,  4.  ,  2.75],
                         [ 6.25,  4.25,  4.25],
                         [ 3.25,  5.  ,  1.75]],

                        [[ 4.75,  1.75,  3.  ],
                         [ 3.  ,  6.25,  7.  ],
                         [ 4.5 ,  5.5 ,  3.25]]],


                       [[[ 4.75,  3.5 ,  2.75],
                         [ 8.75,  3.25,  5.5 ],
                         [ 4.75,  4.  ,  2.75]],

                        [[ 5.25,  1.  ,  5.5 ],
                         [ 5.5 ,  3.25,  2.  ],
                         [ 2.75,  2.75,  7.  ]],

                        [[ 4.75,  5.5 ,  2.75],
                         [ 6.75,  4.75,  4.  ],
                         [ 5.25,  5.25,  5.75]]]])

    grad_pool_out = np.array([[[[6, 5, 9],
                             [2, 4, 0],
                             [9, 0, 2]],

                            [[1, 1, 1],
                             [8, 5, 4],
                             [1, 3, 5]],

                            [[5, 6, 2],
                             [7, 3, 4],
                             [6, 5, 1]]],


                           [[[8, 1, 1],
                             [1, 7, 1],
                             [2, 9, 9]],

                            [[7, 4, 9],
                             [7, 7, 4],
                             [2, 1, 5]],

                            [[2, 0, 1],
                             [6, 1, 8],
                             [1, 7, 2]]],


                           [[[7, 0, 7],
                             [0, 6, 9],
                             [7, 3, 3]],

                            [[9, 6, 5],
                             [6, 4, 7],
                             [3, 3, 5]],

                            [[1, 0, 0],
                             [4, 2, 9],
                             [8, 5, 0]]],


                           [[[8, 4, 7],
                             [8, 7, 2],
                             [1, 3, 1]],

                            [[0, 0, 8],
                             [4, 2, 2],
                             [1, 9, 6]],

                            [[8, 8, 6],
                             [8, 7, 6],
                             [0, 6, 3]]]])

    grad_pool_inp = np.array([[[[ 1.5 ,  1.5 ,  1.25,  1.25,  2.25,  2.25],
                                 [ 1.5 ,  1.5 ,  1.25,  1.25,  2.25,  2.25],
                                 [ 0.5 ,  0.5 ,  1.  ,  1.  ,  0.  ,  0.  ],
                                 [ 0.5 ,  0.5 ,  1.  ,  1.  ,  0.  ,  0.  ],
                                 [ 2.25,  2.25,  0.  ,  0.  ,  0.5 ,  0.5 ],
                                 [ 2.25,  2.25,  0.  ,  0.  ,  0.5 ,  0.5 ]],

                                [[ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                 [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25],
                                 [ 2.  ,  2.  ,  1.25,  1.25,  1.  ,  1.  ],
                                 [ 2.  ,  2.  ,  1.25,  1.25,  1.  ,  1.  ],
                                 [ 0.25,  0.25,  0.75,  0.75,  1.25,  1.25],
                                 [ 0.25,  0.25,  0.75,  0.75,  1.25,  1.25]],

                                [[ 1.25,  1.25,  1.5 ,  1.5 ,  0.5 ,  0.5 ],
                                 [ 1.25,  1.25,  1.5 ,  1.5 ,  0.5 ,  0.5 ],
                                 [ 1.75,  1.75,  0.75,  0.75,  1.  ,  1.  ],
                                 [ 1.75,  1.75,  0.75,  0.75,  1.  ,  1.  ],
                                 [ 1.5 ,  1.5 ,  1.25,  1.25,  0.25,  0.25],
                                 [ 1.5 ,  1.5 ,  1.25,  1.25,  0.25,  0.25]]],


                               [[[ 2.  ,  2.  ,  0.25,  0.25,  0.25,  0.25],
                                 [ 2.  ,  2.  ,  0.25,  0.25,  0.25,  0.25],
                                 [ 0.25,  0.25,  1.75,  1.75,  0.25,  0.25],
                                 [ 0.25,  0.25,  1.75,  1.75,  0.25,  0.25],
                                 [ 0.5 ,  0.5 ,  2.25,  2.25,  2.25,  2.25],
                                 [ 0.5 ,  0.5 ,  2.25,  2.25,  2.25,  2.25]],

                                [[ 1.75,  1.75,  1.  ,  1.  ,  2.25,  2.25],
                                 [ 1.75,  1.75,  1.  ,  1.  ,  2.25,  2.25],
                                 [ 1.75,  1.75,  1.75,  1.75,  1.  ,  1.  ],
                                 [ 1.75,  1.75,  1.75,  1.75,  1.  ,  1.  ],
                                 [ 0.5 ,  0.5 ,  0.25,  0.25,  1.25,  1.25],
                                 [ 0.5 ,  0.5 ,  0.25,  0.25,  1.25,  1.25]],

                                [[ 0.5 ,  0.5 ,  0.  ,  0.  ,  0.25,  0.25],
                                 [ 0.5 ,  0.5 ,  0.  ,  0.  ,  0.25,  0.25],
                                 [ 1.5 ,  1.5 ,  0.25,  0.25,  2.  ,  2.  ],
                                 [ 1.5 ,  1.5 ,  0.25,  0.25,  2.  ,  2.  ],
                                 [ 0.25,  0.25,  1.75,  1.75,  0.5 ,  0.5 ],
                                 [ 0.25,  0.25,  1.75,  1.75,  0.5 ,  0.5 ]]],


                               [[[ 1.75,  1.75,  0.  ,  0.  ,  1.75,  1.75],
                                 [ 1.75,  1.75,  0.  ,  0.  ,  1.75,  1.75],
                                 [ 0.  ,  0.  ,  1.5 ,  1.5 ,  2.25,  2.25],
                                 [ 0.  ,  0.  ,  1.5 ,  1.5 ,  2.25,  2.25],
                                 [ 1.75,  1.75,  0.75,  0.75,  0.75,  0.75],
                                 [ 1.75,  1.75,  0.75,  0.75,  0.75,  0.75]],

                                [[ 2.25,  2.25,  1.5 ,  1.5 ,  1.25,  1.25],
                                 [ 2.25,  2.25,  1.5 ,  1.5 ,  1.25,  1.25],
                                 [ 1.5 ,  1.5 ,  1.  ,  1.  ,  1.75,  1.75],
                                 [ 1.5 ,  1.5 ,  1.  ,  1.  ,  1.75,  1.75],
                                 [ 0.75,  0.75,  0.75,  0.75,  1.25,  1.25],
                                 [ 0.75,  0.75,  0.75,  0.75,  1.25,  1.25]],

                                [[ 0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
                                 [ 0.25,  0.25,  0.  ,  0.  ,  0.  ,  0.  ],
                                 [ 1.  ,  1.  ,  0.5 ,  0.5 ,  2.25,  2.25],
                                 [ 1.  ,  1.  ,  0.5 ,  0.5 ,  2.25,  2.25],
                                 [ 2.  ,  2.  ,  1.25,  1.25,  0.  ,  0.  ],
                                 [ 2.  ,  2.  ,  1.25,  1.25,  0.  ,  0.  ]]],


                               [[[ 2.  ,  2.  ,  1.  ,  1.  ,  1.75,  1.75],
                                 [ 2.  ,  2.  ,  1.  ,  1.  ,  1.75,  1.75],
                                 [ 2.  ,  2.  ,  1.75,  1.75,  0.5 ,  0.5 ],
                                 [ 2.  ,  2.  ,  1.75,  1.75,  0.5 ,  0.5 ],
                                 [ 0.25,  0.25,  0.75,  0.75,  0.25,  0.25],
                                 [ 0.25,  0.25,  0.75,  0.75,  0.25,  0.25]],

                                [[ 0.  ,  0.  ,  0.  ,  0.  ,  2.  ,  2.  ],
                                 [ 0.  ,  0.  ,  0.  ,  0.  ,  2.  ,  2.  ],
                                 [ 1.  ,  1.  ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
                                 [ 1.  ,  1.  ,  0.5 ,  0.5 ,  0.5 ,  0.5 ],
                                 [ 0.25,  0.25,  2.25,  2.25,  1.5 ,  1.5 ],
                                 [ 0.25,  0.25,  2.25,  2.25,  1.5 ,  1.5 ]],

                                [[ 2.  ,  2.  ,  2.  ,  2.  ,  1.5 ,  1.5 ],
                                 [ 2.  ,  2.  ,  2.  ,  2.  ,  1.5 ,  1.5 ],
                                 [ 2.  ,  2.  ,  1.75,  1.75,  1.5 ,  1.5 ],
                                 [ 2.  ,  2.  ,  1.75,  1.75,  1.5 ,  1.5 ],
                                 [ 0.  ,  0.  ,  1.5 ,  1.5 ,  0.75,  0.75],
                                 [ 0.  ,  0.  ,  1.5 ,  1.5 ,  0.75,  0.75]]]])

    return pool_inp, pool_out, grad_pool_out, grad_pool_inp

inp, w, b, grad_out = get_fake_data()
out, grad_inp, grad_w, grad_b = get_conv_answer()

pool_inp, pool_out, grad_pool_out, grad_pool_inp = get_pool_data_answer()

try:
    test_conv_out = conv2d_forward(inp, w, b, 2, 0)
except:
    print('[FAILED] conv2d_forward: bug in codes, can not run for inp.shape = (4, 3, 6, 6), w.shape = (4, 3, 2, 2), ker_size = 2, pad = 0')
else:
    if test_conv_out.shape != out.shape:
        print('[ERROR] conv2d_forward: output shape is not correct')
    else:
        diff = test_conv_out - out
        if abs(diff).max() > 1e-5:
            print('[ERROR] conv2d_forward: output value is not correct')
        else:
            print('[PASS] conv2d_forward: all correct')

flag = 1

try:
    test_grad_inp, test_grad_w, test_grad_b = conv2d_backward(inp, grad_out, w, b, 2, 0)
except:
    print('[FAILED] conv2d_backward: bug in codes, can not run for inp.shape = (4, 3, 6, 6), grad.shape = (4, 4, 7, 7), w.shape = (4, 3, 2, 2), ker_size = 2, pad = 0')
    flag = 0
else:
    if test_grad_inp.shape != grad_inp.shape:
        print('[ERROR] conv2d_backward: grad_input shape is not correct')
        flag = 0
    else:
        diff = test_grad_inp - grad_inp
        if abs(diff).max() > 1e-5:
            print('[ERROR] conv2d_backward: grad_input value is not correct')
            flag = 0

    if test_grad_w.shape != grad_w.shape:
        print('[ERROR] conv2d_backward: grad_w shape is not correct')
        flag = 0
    else:
        diff = test_grad_w - grad_w
        if abs(diff).max() > 1e-5:
            print('[ERROR] conv2d_backward: grad_w value is not correct', abs(diff).max())
            flag = 0

    if test_grad_b.shape != grad_b.shape:
        print('[ERROR] conv2d_backward: grad_b shape is not correct')
        flag = 0
    else:
        diff = test_grad_b - grad_b
        if abs(diff).max() > 1e-5:
            print('[ERROR] conv2d_backward: grad_b value is not correct')
            flag = 0
finally:
    if flag:
        print('[PASS] conv2d_backward: all correct')

try:
    test_pool_out = avgpool2d_forward(pool_inp, 2, 0)
except:
    print('[FAILED] avgpool2d_forward: bug in codes, can not run for inp.shape = (4, 3, 6, 6), ker_size = 2, pad = 0')
else:
    if test_pool_out.shape != pool_out.shape:
        print('[ERROR] avgpool2d_forward: output shape is not correct')
    else:
        diff = test_pool_out - pool_out
        if abs(diff).max() > 1e-5:
            print('[ERROR] avgpool2d_forward: output value is not correct')
        else:
            print('[PASS] avgpool2d_forward: all correct')

try:
    test_grad_pool_inp = avgpool2d_backward(pool_inp, grad_pool_out, 2, 0)
except:
    print('[FAILED] avgpool2d_backward: bug in codes, can not run for inp.shape = (4, 3, 6, 6), grad.shape = (4, 3, 4, 4), ker_size = 2, pad = 0')
else:
    if test_grad_pool_inp.shape != grad_pool_inp.shape:
        print('[ERROR] avgpool2d_backward: grad input shape is not correct')
    else:
        diff = test_grad_pool_inp - grad_pool_inp
        if abs(diff).max() > 1e-5:
            print('[ERROR] avgpool2d_backward: grad input value is not correct')
        else:
            print('[PASS] avgpool2d_backward: all correct')