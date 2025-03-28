import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models


from bisect import bisect_right
import time
import math
from PIL import Image
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/dev/shm/ImageNet/val/', type=str, help='trainset directory')
parser.add_argument('--dataset', default='ImageNet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number workers')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--pretrained', default='/data/winycg/TCM/pretrained_models/resnet50_best.pth.tar', type=str, help='pretrained model path')
parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')


# global hyperparameter set
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)


# Model
print('==> Building model..')

net = models.__dict__[args.arch](num_classes=300).cuda()
checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
model_dict = checkpoint['net']
net.load_state_dict(model_dict)
print('model load successfully')

cudnn.benchmark = True
net.eval()

trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
])

class_name = ['Veronica persica Poir.(阿拉伯婆婆纳)', 'Lysimachia clethroides Duby(矮桃)', 'Alangium chinense (Lour.) Harms(八角枫)', 'Smilax china L.(菝葜)', 'Datura metel L.(白花曼陀罗)', 'Bletilla striata (Thunb. ex A. Murray) Rchb. f.(白及)', 'Fraxinus chinensis Roxb.(白蜡树)', 'Ampelopsis japonica (Thunb.) Makino(白蔹)', 'Chelidonium majus L.(白屈菜)', 'Cynanchum bungei Decne.(白首乌)', 'Pulsatilla chinensis (Bunge) Regel(白头翁)', 'Solanum lyratum\xa0Thunb.(白英)', 'Angelica dahurica\xa0(Fisch. ex Hoffm.) Benth. & Hook. f. ex Franch. & Sav.(白芷)', ' Lilium brownii F. E. Brown ex Miellez var. viridulum Baker(百合)', 'Patrinia scabiosaefolia Fisch.ex Trev.(黄花败酱)', 'Euphorbia maculata L.(斑地锦草)', 'Bothriospermum chinense Bunge(斑种草)', 'Lobelia chinensis Lour.(半边莲)', 'Pinellia ternata (Thunb.) Ten. ex Breitenb.(半夏)', 'Scutellaria barbata D. Don(半枝莲)', 'Mentha canadensis\xa0L.(薄荷)', 'Polygonum aviculare L.(萹蓄)', 'Menispermum dauricum DC.(蝙蝠葛)', 'Areca catechu L.(槟榔)', 'Descurainia sophia\xa0(L.) Webb ex Prantl(播娘蒿)', 'Macleaya cordata (Willd.) R. Br.(博落回)', 'Xanthium strumarium L.(苍耳)', 'Atractylodes lancea (Thunb.) DC.(苍术)', 'Platycladus orientalis (L.) Franco(侧柏)', 'Acorus calamus L.(菖蒲)', 'Hedera nepalensis var. sinensis (Tobler) Rehder(常春藤)', 'Plantago asiatica L.(车前)', 'Penthorum chinense Pursh(扯根菜)', 'Thladiantha dubia Bunge(赤瓟)', 'Clerodendrum bungei Steud.(臭牡丹)', 'Cyathula officinalis K. C. Kuan(川牛膝)', 'Ligusticum chuanxiong Hort.\xa0(川芎)', 'Andrographis paniculata (Burm. f.) Wall. ex Nees in Wallich(穿心莲)', 'Sagittaria trifolia subsp. leucopetala (Miquel) Q. F. Wang(慈姑)', 'Cirsium arvense var. integrifolium Wimm. & Grab.(刺儿菜)', 'Aralia elata (Miq.) Seem.(楤木)', 'Euphorbia pekinensis Rupr.(大戟)', 'Sargentodoxa cuneata (Oliv.) Rehd. & E. H. Wilson in C. S. Sargent(大血藤)', 'Salvia miltiorrhiza Bunge(丹参)', 'Mazus stachydifolius\xa0(Turcz.) Maxim.(弹刀子菜)', 'Angelica sinensis\xa0(Oliv.) Diels(当归)', 'Codonopsis pilosula (Franch.) Nannf.(党参)', 'Lapsanastrum apogonoides (Maxim.) Pak & K.Bremer(稻槎菜)', 'Juncus effusus\xa0L.(灯芯草)', 'Hypericum japonicum Thunb. in Murr.(地耳草)', 'Geranium wilfordii Maxim.(地肤)', 'Rehmannia glutinosa (Gaertn.) Libosch. ex Fisch. & C. A. Mey.(地黄)', 'Lycopus lucidus Turcz. ex Benth.(地笋)', 'Sanguisorba officinalis L.(地榆)', 'Atropa belladonna\xa0L.(颠茄)', 'Androsace umbellata (Lour.) Merr.(点地梅)', 'Sauromatum giganteum (Engl.) Cusimano & Hett.(独角莲)', 'Asarum forbesii Maxim.(杜衡)', 'Eucommia ulmoides Oliv.(杜仲)', 'Crocus sativus L.(番红花)', 'Potentilla discolor Bunge(翻白草)', 'Stellaria media (L.) Vill.(繁缕)', 'Saposhnikovia divaricata (Turcz.) Schischk.(防风)', 'Euphorbia hirta L.(飞扬草)', "Phedimus aizoon (L.) 't Hart(费菜)", 'Stephania tetrandra S. Moore(粉防己)', 'Clinopodium chinense (Benth.) Kuntze(风轮菜)', 'Coniogramme japonica (Thunb.) Diels(凤了蕨)', 'Impatiens balsamina L.(凤仙花)', 'Trigonotis peduncularis (Trevis.) Benth. ex Baker & S. Moore(附地菜)', 'Rubus idaeus\xa0L(覆盆子)', 'Stachys sieboldii\xa0Miq.(甘露子)', 'Periploca sepium Bunge(杠柳)', 'Rhodiola cretinii subsp. sinoalpina (Fröd.) H. Ohba(高山红景天)', 'Pueraria montana var. lobata (Ohwi) Maesen & S. M. Almeida(葛)', 'Ilex cornuta Lindl. & Paxton(枸骨)', 'Lycium chinense Mill.(枸杞)', "Broussonetia papyrifera (L.) L'Hér. ex Vent.(构)", 'Eriocaulon buergerianum Körn.(谷精草)', 'Polygala japonica Houtt.(瓜子金)', 'Alkekengi officinarum var. franchetii (Mast.) R. J. Wang(挂金灯)', 'Hypericum perforatum L.(贯叶连翘)', 'Cyrtomium fortunei J. Sm.(贯众)', 'Vicia cracca L.(广布野豌豆)', 'Bidens pilosa L.(鬼针草)', 'Lysimachia christinae Hance(过路黄)', 'Lygodium japonicum (Thunb.) Sw.(海金沙)', 'Pittosporum tobira (Thunb.) W. T. Aiton(海桐)', 'Clerodendrum trichotomum Thunb.(海州常山)', 'Mimosa pudica L.(含羞草)', 'Scutellaria indica L.(韩信草)', 'Rorippa indica\xa0(L.) Hiern(蔊菜)', 'Albizia julibrissin Durazz.(合欢)', 'Pleuropterus multiflorus (Thunb.) Nakai(何首乌)', 'Taxus wallichiana var. chinensis (Pilg.) Florin(红豆杉)', 'Carthamus tinctorius L.(红花)', 'Oxalis corymbosa DC.(红花酢浆草)', "Lycoris aurea (L'Hér.) Herb.(忽地笑)", 'Drynaria roosii Nakaike(槲蕨)', 'Saxifraga stolonifera Curtis(虎耳草)', 'Reynoutria japonica Houtt.(虎杖)', 'Styphnolobium japonicum (L.) Schott(槐)', 'Youngia japonica (L.) DC. (黄鹌菜)', 'Hemerocallis citrina Baroni(黄花菜)', 'Vitex negundo L.(黄荆)', 'Polygonatum sibiricum Redouté(黄精)', 'Berberis amurensis Rupr.(黄芦木)', 'Scutellaria baicalensis Georgi(黄芩)', 'Ranunculus chinensis Bunge(茴茴蒜)', 'Glechoma longituba (Nakai) Kupr.(活血丹)', 'Sedum sarmentosum Bunge(垂盆草)', 'Agastache rugosa (Fisch. & C. A. Mey.) Kuntze(藿香)', 'Rhodotypos scandens\xa0(Thunb.) Makino(鸡麻)', 'Paederia  foetida L.(鸡屎藤)', 'Viola acuminata Ledeb.(鸡腿堇菜)', 'Chloranthus serratus (Thunb.) Roem. & Schult.(及己)', 'Houttuynia cordata\xa0Thunb.(蕺菜)', 'Cirsium japonicum Fisch. ex DC.(蓟)', 'Nerium oleander L.(夹竹桃)', 'Crepidiastrum sonchifolium (Bunge) Pak & Kawano(尖裂假还阳参)', 'Gynostemma pentaphyllum (Thunb.) Makino(绞股蓝)', 'Sambucus javanica Reinw. ex Blume(接骨草)', 'Sambucus williamsii Hance(接骨木)', 'Cuscuta japonica Choisy in Zoll.(金灯藤)', 'Stephania cephalantha Hayata(金线吊乌龟)', 'Rosa laevigata Michx.(金樱子)', 'Lysimachia grammica Hance(金爪儿)', 'Ajuga decumbens Thunb.(筋骨草)', 'Nepeta cataria L.(荆芥)', 'Vicia sativa L.(救荒野豌豆)', 'Platycodon grandiflorus (Jacq.) A. DC.(桔梗)', 'Chrysanthemum morifolium Ramat.(菊花)', 'Gynura japonica (Thunb.) Juel(菊三七)', 'Selaginella tamariscina (P. Beauv.) Spring(卷柏)', 'Senna tora (L.) Roxb.(决明)', 'Justicia procumbens L.(爵床)', 'Persicaria perfoliata (L.) H. Gross(扛板归)', 'Corydalis incisa (Thunb.) Pers.(刻叶紫堇)', 'Sophora flavescens Aiton(苦参)', 'Trichosanthes kirilowii Maxim.(栝楼)', 'Geranium wilfordii Maxim.(老鹳草)', 'Amana edulis (Miq.) Honda(老鸦瓣)', 'Viola japonica Langsd. ex DC.(犁头草)', 'Chenopodium album L.(藜)', 'Veratrum nigrum\xa0L.(藜芦)', 'Eclipta prostrata (L.) L.(鳢肠)', 'Salvia plebeia\xa0R. Br.(荔枝草)', 'Forsythia suspensa (Thunb.) Vahl(连翘)', 'Hyoscyamus niger\xa0L.(莨菪)', ' Lysimachia congestiflora Hemsl.(临时救)', 'Solanum nigrum L.(龙葵)', 'Agrimonia pilosa Ledeb.(龙牙草)', 'Adenophora tetraphylla (Thunb.) Fisch.(轮叶沙参)', 'Apocynum venetum L.(罗布麻)', 'Raphanus sativus L.(萝卜)', 'Cynanchum rostellatum (Turcz.) Liede & Khanum(萝藦)', 'Trachelospermum jasminoides (Lindl.) Lem.(络石)', 'Astilbe chinensis (Maxim.) Franch. & Sav.(落新妇)', 'Humulus scandens (Lour.) Merr.(葎草)', 'Verbena officinalis L.(马鞭草)', 'Zehneria indica (Lour.)Keraudren(马㼎儿)', 'Portulaca oleracea L.(马齿苋)', 'Aristolochia debilis Siebold & Zucc.(马兜铃)', 'Baphicacanthus cusia(Nees)Bremek.(马蓝)', 'Ophiopogon japonicus\xa0(L. f.) Ker Gawl.(麦冬)', 'Vaccaria segetalis(Neck.)Garcke(麦蓝菜)', 'Ranunculus ternatus\xa0Thunb.(猫爪草)', 'Ranunculus japonicus Thunb.(毛茛)', 'Spatholobus suberectus Dunn(密花豆)', 'Vitex negundo var. cannabifolia (Siebold & Zucc.) Hand.-Mazz.(牡荆)', 'Momordica cochinchinensis (Lour.) Spreng.(木鳖子)', 'Cocculus orbiculatus (L.) DC.(木防己)', 'Oroxylum indicum (L.) Kurz(木蝴蝶)', 'Hibiscus syriacus L.(木槿)', 'Equisetum hyemale L.(木贼)', 'Nandina domestica Thunb.(南天竹)', 'Arctium lappa L.(牛蒡)', 'Achyranthes bidentata\xa0Blume(牛膝)', 'Origanum vulgare L.(牛至)', 'Ligustrum lucidum W. T. Aiton(女贞)', 'Eupatorium fortunei\xa0Turcz.(佩兰)', 'Eriobotrya japonica (Thunb.) Lindl.(枇杷)', 'Sinosenecio oldhamianus (Maxim.) B. Nord.(蒲儿根)', 'Taraxacum mongolicum Hand.-Mazz.(蒲公英)', 'Viola diffusa Ging. in DC.(七星莲)', 'Paris polyphylla Sm.(七叶一枝花)', 'Capsella bursa-pastoris (L.) Medik.(荠)', 'Adenophora trachelioides Maxim.(荠苨)', 'Senecio scandens Buch.-Ham. ex D. Don(千里光)', 'Ipomoea nil (L.) Roth(牵牛)', 'Peucedanum praeruptorum Dunn(前胡)', 'Euryale ferox Salisb. ex K. D. Koenig & Sims(芡)', 'Rubia  cordifolia L.(茜草)', 'Fagopyrum esculentum Moench(荞麦)', 'Cardiocrinum cathayanum (E. H. Wilson) Stearn(荞麦叶大百合)', 'Torilis scabra (Thunb.) DC.(窃衣)', 'Abutilon theophrasti Medikus(苘麻)', 'Dianthus superbus L. (瞿麦)', 'P.bistorta L.(拳参)', 'Panax ginseng C. A. Mey.(人参)', 'Lonicera japonica Thunb.(忍冬)', 'Euphorbia esula L.(乳浆大戟)', 'Saururus chinensis (Lour.) Baill.(三白草)', 'Coptis deltoidea C. Y. Cheng & P. G. Xiao(三角叶黄连)', 'Panax notoginseng (Burkill) F. H. Chen ex C. H. Chow(三七)', 'Morus alba\xa0L.(桑)', 'Cornus officinalis Sieb. & Zucc.(山茱萸)', 'Glehnia littoralis F. Schmidt ex Miq.(珊瑚菜)', 'Solanum pseudocapsicum\xa0L.(珊瑚樱)', 'Phytolacca acinosa Roxb.(商陆)', 'Paeonia lactiflora Pall.(芍药)', 'Corydalis ophiocarpa Hook. f. & Thomson(蛇果黄堇)', 'Belamcanda chinensis\xa0(L.) Redouté(射干)', 'Actaea cimicifuga L.(升麻)', 'Staphylea bumalda DC.(省沽油)', 'Mahonia fortunei (Lindl.) Fedde(十大功劳)', 'Centipeda minima (L.) A. Braun & Asch.(石胡荽)', 'Ranunculus sceleratus L.(石龙芮)', "Lycoris radiata (L'Hér.) Herb.(石蒜)", 'Pyrrosia lingua (Thunb.) Farw.(石韦)', 'Dianthus chinensis L.(石竹)', 'Pseudognaphalium affine (D. Don) Anderb.(鼠曲草)', 'Dioscorea polystachya Turcz.(薯蓣)', 'Persicaria hydropiper (L.) Spach(水蓼)', 'Isatis tinctoria L.(菘蓝)', 'Cycas revoluta Thunb.(苏铁)', 'Isodon rubescens (Hemsl.) H. Hara(碎米桠)', 'Semiaquilegia adoxoides (DC.) Makino(天葵)', 'Gastrodia elata Bl.(天麻)', 'Asparagus cochinchinensis (Lour.) Merr.(天门冬)', 'Carpesium abrotanoides L.(天名精)', 'Arisaema heterophyllum Blume(天南星)', 'Convolvulus arvensis L.(田旋花)', 'Acalypha australis L.(铁苋菜)', 'Clematis florida Thunb.(铁线莲)', 'Draba nemorosa L.(葶苈)', 'Mazus pumilus (Burm. f.) Steenis(通泉草)', 'Tetrapanax papyrifer (Hook.) K. Koch(通脱木)', 'Pilea pumila (L.) A. Gray(透茎冷水花)', 'Talinum paniculatum (Jacq.) Gaertn.(土人参)', 'Syneilesis aconitifolia\xa0(Bunge) Maxim.(兔儿伞)', 'Cuscuta chinensis Lam.(菟丝子)', 'Orostachys fimbriata (Turcz.) A. Berger(瓦松)', 'Lepisorus thunbergianus (Kaulf.) Ching(瓦韦)', 'Euonymus alatus (Thunb.) Siebold(卫矛)', 'Equisetum arvense L.(问荆)', 'Triadica sebifera (L.) Small(乌桕)', 'Aconitum carmichaelii Debeaux(乌头)', 'Schisandra chinensis (Turcz.) Baill.(五味子)', 'Thlaspi arvense L.(菥蓂)', ' Siegesbeckia orientalis L(豨莶)', 'Asarum heterotropoides F. Schmidt(细辛)', 'Eleutherococcus nodiflorus (Dunn) S. Y. Hu(细柱五加)', 'Bupleurum scorzonerifolium Willd.(狭叶柴胡)', 'Prunella vulgaris\xa0L.(夏枯草)', 'Curculigo orchioides Gaertn.(仙茅)', 'Cyperus rotundus L.(香附子)', 'Vicia hirsuta (L.) Gray(小巢菜)', 'Allium macrostemon Bunge(小根蒜)', 'Vincetoxicum pycnostelma Kitag.(徐长卿)', 'Scrophularia ningpoensis\xa0Hemsl.(玄参)', 'Inula japonica Thunb.(旋覆花)', 'Commelina communis L.(鸭跖草)', 'Corydalis yanhusuo (Y. H. Chou & C. C. Hsu) W. T. Wang ex Z. Y. Su & C. Y. Wu(延胡索)', 'Rhus chinensis Mill.(盐麸木)', 'Codonopsis lanceolata (Siebold & Zucc.) Trautv.(羊乳)', 'Daucus carota L.(野胡萝卜)', 'Erigeron annuus (L.) Pers.(一年蓬)', 'Leonurus japonicus Houtt.(益母草)', 'Coix lacryma-jobi L.(薏苡)', 'Pteroxygonum giraldii\xa0Damm. & Diels(翼蓼)', 'Sceptridium ternatum (Thunb.) Lyon(阴地蕨)', 'Siphonostegia chinensis Benth.(阴行草)', 'Artemisia capillaris\xa0Thunb.(茵陈蒿)', 'Ginkgo biloba L.(银杏)', 'Epimedium brevicornu\xa0Maxim.(淫羊藿)', 'Papaver somniferum L.(罂粟)', 'Yulania denudata (Desr.) D. L. Fu(玉兰)', 'Polygonatum odoratum (Mill.) Druce(玉竹)', 'Iris tectorum Maxim.(鸢尾)', 'Hypericum sampsonii Hance(元宝草)', 'Daphne genkwa Siebold & Zucc.(芫花)', 'Polygala tenuifolia Willd.(远志)', 'Gleditsia sinensis\xa0Lam.(皂荚)', 'Euphorbia helioscopia L.(泽漆)', 'Alisma plantago-aquatica L.(泽泻)', 'Catharanthus roseus (L.) G. Don(长春花)', 'Fritillaria thunbergii Miq.(浙贝母)', 'Anemarrhena asphodeloides\xa0Bunge(知母)', 'Gardenia  jasminoides J. Ellis(栀子)', 'Ixeris chinensis (Thunb.) Nakai(中华苦荬菜)', 'Rumex crispus L.(皱叶酸模)', 'Orychophragmus violaceus (L.) O. E. Schulz(诸葛菜)', 'Lithospermum zollingeri A. DC.(梓木草)', 'Viola philippica\xa0Cav.(紫花地丁)', 'Angelica decursiva\xa0(Miq.) Franch. & Sav.(紫花前胡)', 'Osmunda japonica Thunb.(紫萁)', 'Perilla frutescens (L.) Britton(紫苏)', 'Aster tataricus L. f.(紫菀)', 'Callicarpa bodinieri H. Lév.(紫珠)']

with torch.no_grad():
    raw_image = Image.open(args.image_path).convert("RGB")
    image = trans(raw_image)
    image = torch.unsqueeze(image, 0).cuda()

    logits = net(image)
    probs = F.softmax(logits, dim=1)
    value, index = probs.max(dim=1)
    print('Predicted category is '+class_name[index.item()]+'.')



