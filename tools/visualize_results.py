import os
import copy
import numpy as np
import cv2
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt
import torch

import mmcv
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes


cali_split_100 = ['scene-0397', 'scene-0219', 'scene-0295', 'scene-0354', 'scene-0214', 'scene-0154', 'scene-0984', 'scene-0656', 'scene-0640', 'scene-0246', 'scene-0202', 'scene-0600', 'scene-0509', 'scene-0241', 'scene-0809', 'scene-0250', 'scene-0200', 'scene-0204', 'scene-0525', 'scene-0686', 'scene-0752', 'scene-0711', 'scene-0714', 'scene-0515', 'scene-1009', 'scene-0403', 'scene-0362', 'scene-0718', 'scene-0369', 'scene-0812', 'scene-0585', 'scene-0432', 'scene-0545', 'scene-0735', 'scene-0179', 'scene-0679', 'scene-0850', 'scene-1105', 'scene-0426', 'scene-1094', 'scene-0053', 'scene-0434', 'scene-0894', 'scene-0736', 'scene-0568', 'scene-0052', 'scene-0127', 'scene-1021', 'scene-0025', 'scene-0163', 'scene-0546', 'scene-0704', 'scene-0042', 'scene-0453', 'scene-0504', 'scene-0383', 'scene-0668', 'scene-0897', 'scene-1011', 'scene-1088', 'scene-0673', 'scene-1104', 'scene-0696', 'scene-0813', 'scene-0958', 'scene-0253', 'scene-0805', 'scene-0541', 'scene-0677', 'scene-0878', 'scene-0175', 'scene-0733', 'scene-0685', 'scene-0456', 'scene-1044', 'scene-0057', 'scene-1024', 'scene-0066', 'scene-0323', 'scene-0855', 'scene-0650', 'scene-0688', 'scene-0896', 'scene-0418', 'scene-0239', 'scene-0655', 'scene-0474', 'scene-0697', 'scene-0511', 'scene-1045', 'scene-0759', 'scene-0901', 'scene-0152', 'scene-1084', 'scene-1098', 'scene-0415', 'scene-0592', 'scene-0674', 'scene-0598', 'scene-0292']

train_rain = ["scene-0440", "scene-0441", "scene-0442", "scene-0443", "scene-0444", "scene-0445", "scene-0446", "scene-0447", "scene-0448", "scene-0449", "scene-0450", "scene-0451", "scene-0452", "scene-0453", "scene-0454", "scene-0455", "scene-0456", "scene-0457", "scene-0458", "scene-0459", "scene-0461", "scene-0462", "scene-0463", "scene-0464", "scene-0465", "scene-0467", "scene-0468", "scene-0469", "scene-0471", "scene-0472", "scene-0474", "scene-0475", "scene-0476", "scene-0477", "scene-0478", "scene-0479", "scene-0480", "scene-0566", "scene-0568", "scene-0570", "scene-0571", "scene-0572", "scene-0573", "scene-0574", "scene-0575", "scene-0576", "scene-0577", "scene-0578", "scene-0580", "scene-0582", "scene-0583", "scene-0584", "scene-0585", "scene-0586", "scene-0587", "scene-0588", "scene-0589", "scene-0590", "scene-0591", "scene-0592", "scene-0593", "scene-0594", "scene-0595", "scene-0596", "scene-0597", "scene-0598", "scene-0599", "scene-0600", "scene-0639", "scene-0640", "scene-0641", "scene-0642", "scene-0643", "scene-0644", "scene-0645", "scene-0647", "scene-0648", "scene-0649", "scene-0650", "scene-0651", "scene-0652", "scene-0804", "scene-0806", "scene-0808", "scene-0810", "scene-0811", "scene-0813", "scene-0815", "scene-0816", "scene-0819", "scene-0820", "scene-0822", "scene-0868", "scene-0869", "scene-0870", "scene-0871", "scene-0872", "scene-0873", "scene-0875", "scene-0876", "scene-0877", "scene-0878", "scene-0880", "scene-0882", "scene-0883", "scene-0884", "scene-0885", "scene-0886", "scene-0887", "scene-0888", "scene-0889", "scene-0890", "scene-0891", "scene-0892", "scene-0893", "scene-0894", "scene-0895", "scene-0896", "scene-0897", "scene-0898", "scene-0899", "scene-0900", "scene-0901", "scene-0902", "scene-0903", "scene-1053", "scene-1074", "scene-1081", "scene-1094", "scene-1095", "scene-1101", "scene-1102", "scene-1104", "scene-1106", "scene-1107", "scene-1108", "scene-1109", "scene-1110", ]
train_sunny = ["scene-0062", "scene-0063", "scene-0064", "scene-0065", "scene-0066", "scene-0067", "scene-0068", "scene-0069", "scene-0070", "scene-0071", "scene-0072", "scene-0073", "scene-0074", "scene-0075", "scene-0076", "scene-0161", "scene-0162", "scene-0163", "scene-0164", "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170", "scene-0171", "scene-0172", "scene-0173", "scene-0174", "scene-0175", "scene-0176", "scene-0199", "scene-0200", "scene-0202", "scene-0203", "scene-0204", "scene-0206", "scene-0207", "scene-0208", "scene-0209", "scene-0210", "scene-0211", "scene-0212", "scene-0213", "scene-0214", "scene-0218", "scene-0219", "scene-0220", "scene-0222", "scene-0224", "scene-0225", "scene-0226", "scene-0227", "scene-0228", "scene-0229", "scene-0230", "scene-0231", "scene-0232", "scene-0233", "scene-0234", "scene-0235", "scene-0236", "scene-0237", "scene-0238", "scene-0239", "scene-0240", "scene-0241", "scene-0242", "scene-0243", "scene-0244", "scene-0245", "scene-0246", "scene-0247", "scene-0248", "scene-0249", "scene-0250", "scene-0251", "scene-0252", "scene-0253", "scene-0254", "scene-0255", "scene-0256", "scene-0257", "scene-0258", "scene-0259", "scene-0260", "scene-0261", "scene-0262", "scene-0263", "scene-0264", "scene-0283", "scene-0284", "scene-0285", "scene-0286", "scene-0287", "scene-0288", "scene-0289", "scene-0290", "scene-0291", "scene-0292", "scene-0293", "scene-0294", "scene-0295", "scene-0296", "scene-0297", "scene-0298", "scene-0299", "scene-0300", "scene-0301", "scene-0302", "scene-0303", "scene-0304", "scene-0305", "scene-0306", "scene-0321", "scene-0323", "scene-0324", "scene-0328", "scene-0388", "scene-0389", "scene-0390", "scene-0391", "scene-0392", "scene-0393", "scene-0394", "scene-0395", "scene-0396", "scene-0397", "scene-0398", "scene-0499", "scene-0500", "scene-0501", "scene-0502", "scene-0504", "scene-0505", "scene-0506", "scene-0507", "scene-0508", "scene-0509", "scene-0510", "scene-0511", "scene-0512", "scene-0513", "scene-0514", "scene-0515", "scene-0517", "scene-0518", "scene-0525", "scene-0526", "scene-0527", "scene-0528", "scene-0529", "scene-0530", "scene-0531", "scene-0532", "scene-0533", "scene-0534", "scene-0535", "scene-0536", "scene-0537", "scene-0538", "scene-0539", "scene-0541", "scene-0542", "scene-0543", "scene-0544", "scene-0545", "scene-0546", "scene-0646", "scene-0653", "scene-0654", "scene-0655", "scene-0656", "scene-0657", "scene-0658", "scene-0659", "scene-0660", "scene-0661", "scene-0662", "scene-0663", "scene-0664", "scene-0665", "scene-0666", "scene-0667", "scene-0668", "scene-0669", "scene-0670", "scene-0671", "scene-0672", "scene-0673", "scene-0674", "scene-0675", "scene-0676", "scene-0677", "scene-0678", "scene-0679", "scene-0681", "scene-0683", "scene-0684", "scene-0685", "scene-0686", "scene-0687", "scene-0688", "scene-0689", "scene-0695", "scene-0696", "scene-0697", "scene-0698", "scene-0700", "scene-0701", "scene-0703", "scene-0704", "scene-0705", "scene-0706", "scene-0707", "scene-0708", "scene-0709", "scene-0710", "scene-0711", "scene-0712", "scene-0713", "scene-0714", "scene-0715", "scene-0716", "scene-0717", "scene-0718", "scene-0719", "scene-0726", "scene-0727", "scene-0728", "scene-0730", "scene-0731", "scene-0733", "scene-0734", "scene-0735", "scene-0736", "scene-0737", "scene-0738", "scene-0739", "scene-0740", "scene-0741", "scene-0744", "scene-0746", "scene-0747", "scene-0749", "scene-0750", "scene-0751", "scene-0752", "scene-0757", "scene-0758", "scene-0759", "scene-0760", "scene-0761", "scene-0762", "scene-0763", "scene-0764", "scene-0765", "scene-0767", "scene-0768", "scene-0769", "scene-0803", "scene-0805", "scene-0809", "scene-0812", "scene-0817", "scene-0821", "scene-0399", "scene-0400", "scene-0401", "scene-0402", "scene-0403", "scene-0405", "scene-0406", "scene-0407", "scene-0408", "scene-0410", "scene-0411", "scene-0412", "scene-0413", "scene-0414", "scene-0415", "scene-0416", "scene-0417", "scene-0418", "scene-0419", "scene-0001", "scene-0002", "scene-0004", "scene-0005", "scene-0006", "scene-0007", "scene-0008", "scene-0009", "scene-0010", "scene-0011", "scene-0019", "scene-0020", "scene-0021", "scene-0022", "scene-0023", "scene-0024", "scene-0025", "scene-0026", "scene-0027", "scene-0028", "scene-0029", "scene-0030", "scene-0031", "scene-0032", "scene-0033", "scene-0034", "scene-0041", "scene-0042", "scene-0043", "scene-0044", "scene-0045", "scene-0046", "scene-0047", "scene-0048", "scene-0049", "scene-0050", "scene-0051", "scene-0052", "scene-0053", "scene-0054", "scene-0055", "scene-0056", "scene-0057", "scene-0058", "scene-0059", "scene-0060", "scene-0061", "scene-0120", "scene-0121", "scene-0122", "scene-0123", "scene-0124", "scene-0125", "scene-0126", "scene-0127", "scene-0128", "scene-0129", "scene-0130", "scene-0131", "scene-0132", "scene-0133", "scene-0134", "scene-0135", "scene-0138", "scene-0139", "scene-0149", "scene-0150", "scene-0151", "scene-0152", "scene-0154", "scene-0155", "scene-0157", "scene-0158", "scene-0159", "scene-0160", "scene-0190", "scene-0191", "scene-0192", "scene-0193", "scene-0194", "scene-0195", "scene-0196", "scene-0347", "scene-0348", "scene-0349", "scene-0350", "scene-0351", "scene-0352", "scene-0353", "scene-0354", "scene-0355", "scene-0356", "scene-0357", "scene-0358", "scene-0359", "scene-0360", "scene-0361", "scene-0362", "scene-0363", "scene-0364", "scene-0365", "scene-0366", "scene-0367", "scene-0368", "scene-0369", "scene-0370", "scene-0371", "scene-0372", "scene-0373", "scene-0374", "scene-0375", "scene-0376", "scene-0377", "scene-0378", "scene-0379", "scene-0380", "scene-0381", "scene-0382", "scene-0383", "scene-0384", "scene-0385", "scene-0386", "scene-0945", "scene-0947", "scene-0949", "scene-0952", "scene-0953", "scene-0955", "scene-0956", "scene-0957", "scene-0958", "scene-0959", "scene-0960", "scene-0961", "scene-0975", "scene-0976", "scene-0977", "scene-0978", "scene-0979", "scene-0980", "scene-0981", "scene-0982", "scene-0983", "scene-0984", "scene-0988", "scene-0989", "scene-0990", "scene-0991", "scene-0177", "scene-0178", "scene-0179", "scene-0180", "scene-0181", "scene-0182", "scene-0183", "scene-0184", "scene-0185", "scene-0187", "scene-0188", "scene-0315", "scene-0316", "scene-0317", "scene-0318", "scene-0420", "scene-0421", "scene-0422", "scene-0423", "scene-0424", "scene-0425", "scene-0426", "scene-0427", "scene-0428", "scene-0429", "scene-0430", "scene-0431", "scene-0432", "scene-0433", "scene-0434", "scene-0435", "scene-0436", "scene-0437", "scene-0438", "scene-0439", "scene-0786", "scene-0787", "scene-0789", "scene-0790", "scene-0791", "scene-0792", "scene-0847", "scene-0848", "scene-0849", "scene-0850", "scene-0851", "scene-0852", "scene-0853", "scene-0854", "scene-0855", "scene-0856", "scene-0858", "scene-0860", "scene-0861", "scene-0862", "scene-0863", "scene-0864", "scene-0865", "scene-0866", "scene-1044", "scene-1045", "scene-1046", "scene-1047", "scene-1048", "scene-1049", "scene-1050", "scene-1051", "scene-1052", "scene-1054", "scene-1055", "scene-1056", "scene-1057", "scene-1058", "scene-1075", "scene-1076", "scene-1077", "scene-1078", "scene-1079", "scene-1080", "scene-1082", "scene-1083", "scene-1084", "scene-1085", "scene-1086", "scene-1087", "scene-1088", "scene-1089", "scene-1090", "scene-1091", "scene-1092", "scene-1093", "scene-1096", "scene-1097", "scene-1098", "scene-1099", "scene-1100", "scene-1105", "scene-0992", "scene-0994", "scene-0995", "scene-0996", "scene-0997", "scene-0998", "scene-0999", "scene-1000", "scene-1001", "scene-1002", "scene-1003", "scene-1004", "scene-1005", "scene-1006", "scene-1007", "scene-1008", "scene-1009", "scene-1010", "scene-1011", "scene-1012", "scene-1013", "scene-1014", "scene-1015", "scene-1016", "scene-1017", "scene-1018", "scene-1019", "scene-1020", "scene-1021", "scene-1022", "scene-1023", "scene-1024", "scene-1025"]

train_night = ["scene-1053", "scene-1074", "scene-1081", "scene-1094", "scene-1095", "scene-1101", "scene-1102", "scene-1104", "scene-1106", "scene-1107", "scene-1108", "scene-1109", "scene-1110", "scene-1044", "scene-1045", "scene-1046", "scene-1047", "scene-1048", "scene-1049", "scene-1050", "scene-1051", "scene-1052", "scene-1054", "scene-1055", "scene-1056", "scene-1057", "scene-1058", "scene-1075", "scene-1076", "scene-1077", "scene-1078", "scene-1079", "scene-1080", "scene-1082", "scene-1083", "scene-1084", "scene-1085", "scene-1086", "scene-1087", "scene-1088", "scene-1089", "scene-1090", "scene-1091", "scene-1092", "scene-1093", "scene-1096", "scene-1097", "scene-1098", "scene-1099", "scene-1100", "scene-1105", "scene-0992", "scene-0994", "scene-0995", "scene-0996", "scene-0997", "scene-0998", "scene-0999", "scene-1000", "scene-1001", "scene-1002", "scene-1003", "scene-1004", "scene-1005", "scene-1006", "scene-1007", "scene-1008", "scene-1009", "scene-1010", "scene-1011", "scene-1012", "scene-1013", "scene-1014", "scene-1015", "scene-1016", "scene-1017", "scene-1018", "scene-1019", "scene-1020", "scene-1021", "scene-1022", "scene-1023", "scene-1024", "scene-1025"]
train_day = ["scene-0440", "scene-0441", "scene-0442", "scene-0443", "scene-0444", "scene-0445", "scene-0446", "scene-0447", "scene-0448", "scene-0449", "scene-0450", "scene-0451", "scene-0452", "scene-0453", "scene-0454", "scene-0455", "scene-0456", "scene-0457", "scene-0458", "scene-0459", "scene-0461", "scene-0462", "scene-0463", "scene-0464", "scene-0465", "scene-0467", "scene-0468", "scene-0469", "scene-0471", "scene-0472", "scene-0474", "scene-0475", "scene-0476", "scene-0477", "scene-0478", "scene-0479", "scene-0480", "scene-0566", "scene-0568", "scene-0570", "scene-0571", "scene-0572", "scene-0573", "scene-0574", "scene-0575", "scene-0576", "scene-0577", "scene-0578", "scene-0580", "scene-0582", "scene-0583", "scene-0584", "scene-0585", "scene-0586", "scene-0587", "scene-0588", "scene-0589", "scene-0590", "scene-0591", "scene-0592", "scene-0593", "scene-0594", "scene-0595", "scene-0596", "scene-0597", "scene-0598", "scene-0599", "scene-0600", "scene-0639", "scene-0640", "scene-0641", "scene-0642", "scene-0643", "scene-0644", "scene-0645", "scene-0647", "scene-0648", "scene-0649", "scene-0650", "scene-0651", "scene-0652", "scene-0804", "scene-0806", "scene-0808", "scene-0810", "scene-0811", "scene-0813", "scene-0815", "scene-0816", "scene-0819", "scene-0820", "scene-0822", "scene-0868", "scene-0869", "scene-0870", "scene-0871", "scene-0872", "scene-0873", "scene-0875", "scene-0876", "scene-0877", "scene-0878", "scene-0880", "scene-0882", "scene-0883", "scene-0884", "scene-0885", "scene-0886", "scene-0887", "scene-0888", "scene-0889", "scene-0890", "scene-0891", "scene-0892", "scene-0893", "scene-0894", "scene-0895", "scene-0896", "scene-0897", "scene-0898", "scene-0899", "scene-0900", "scene-0901", "scene-0902", "scene-0903", "scene-0062", "scene-0063", "scene-0064", "scene-0065", "scene-0066", "scene-0067", "scene-0068", "scene-0069", "scene-0070", "scene-0071", "scene-0072", "scene-0073", "scene-0074", "scene-0075", "scene-0076", "scene-0161", "scene-0162", "scene-0163", "scene-0164", "scene-0165", "scene-0166", "scene-0167", "scene-0168", "scene-0170", "scene-0171", "scene-0172", "scene-0173", "scene-0174", "scene-0175", "scene-0176", "scene-0199", "scene-0200", "scene-0202", "scene-0203", "scene-0204", "scene-0206", "scene-0207", "scene-0208", "scene-0209", "scene-0210", "scene-0211", "scene-0212", "scene-0213", "scene-0214", "scene-0218", "scene-0219", "scene-0220", "scene-0222", "scene-0224", "scene-0225", "scene-0226", "scene-0227", "scene-0228", "scene-0229", "scene-0230", "scene-0231", "scene-0232", "scene-0233", "scene-0234", "scene-0235", "scene-0236", "scene-0237", "scene-0238", "scene-0239", "scene-0240", "scene-0241", "scene-0242", "scene-0243", "scene-0244", "scene-0245", "scene-0246", "scene-0247", "scene-0248", "scene-0249", "scene-0250", "scene-0251", "scene-0252", "scene-0253", "scene-0254", "scene-0255", "scene-0256", "scene-0257", "scene-0258", "scene-0259", "scene-0260", "scene-0261", "scene-0262", "scene-0263", "scene-0264", "scene-0283", "scene-0284", "scene-0285", "scene-0286", "scene-0287", "scene-0288", "scene-0289", "scene-0290", "scene-0291", "scene-0292", "scene-0293", "scene-0294", "scene-0295", "scene-0296", "scene-0297", "scene-0298", "scene-0299", "scene-0300", "scene-0301", "scene-0302", "scene-0303", "scene-0304", "scene-0305", "scene-0306", "scene-0321", "scene-0323", "scene-0324", "scene-0328", "scene-0388", "scene-0389", "scene-0390", "scene-0391", "scene-0392", "scene-0393", "scene-0394", "scene-0395", "scene-0396", "scene-0397", "scene-0398", "scene-0499", "scene-0500", "scene-0501", "scene-0502", "scene-0504", "scene-0505", "scene-0506", "scene-0507", "scene-0508", "scene-0509", "scene-0510", "scene-0511", "scene-0512", "scene-0513", "scene-0514", "scene-0515", "scene-0517", "scene-0518", "scene-0525", "scene-0526", "scene-0527", "scene-0528", "scene-0529", "scene-0530", "scene-0531", "scene-0532", "scene-0533", "scene-0534", "scene-0535", "scene-0536", "scene-0537", "scene-0538", "scene-0539", "scene-0541", "scene-0542", "scene-0543", "scene-0544", "scene-0545", "scene-0546", "scene-0646", "scene-0653", "scene-0654", "scene-0655", "scene-0656", "scene-0657", "scene-0658", "scene-0659", "scene-0660", "scene-0661", "scene-0662", "scene-0663", "scene-0664", "scene-0665", "scene-0666", "scene-0667", "scene-0668", "scene-0669", "scene-0670", "scene-0671", "scene-0672", "scene-0673", "scene-0674", "scene-0675", "scene-0676", "scene-0677", "scene-0678", "scene-0679", "scene-0681", "scene-0683", "scene-0684", "scene-0685", "scene-0686", "scene-0687", "scene-0688", "scene-0689", "scene-0695", "scene-0696", "scene-0697", "scene-0698", "scene-0700", "scene-0701", "scene-0703", "scene-0704", "scene-0705", "scene-0706", "scene-0707", "scene-0708", "scene-0709", "scene-0710", "scene-0711", "scene-0712", "scene-0713", "scene-0714", "scene-0715", "scene-0716", "scene-0717", "scene-0718", "scene-0719", "scene-0726", "scene-0727", "scene-0728", "scene-0730", "scene-0731", "scene-0733", "scene-0734", "scene-0735", "scene-0736", "scene-0737", "scene-0738", "scene-0739", "scene-0740", "scene-0741", "scene-0744", "scene-0746", "scene-0747", "scene-0749", "scene-0750", "scene-0751", "scene-0752", "scene-0757", "scene-0758", "scene-0759", "scene-0760", "scene-0761", "scene-0762", "scene-0763", "scene-0764", "scene-0765", "scene-0767", "scene-0768", "scene-0769", "scene-0803", "scene-0805", "scene-0809", "scene-0812", "scene-0817", "scene-0821", "scene-0399", "scene-0400", "scene-0401", "scene-0402", "scene-0403", "scene-0405", "scene-0406", "scene-0407", "scene-0408", "scene-0410", "scene-0411", "scene-0412", "scene-0413", "scene-0414", "scene-0415", "scene-0416", "scene-0417", "scene-0418", "scene-0419", "scene-0001", "scene-0002", "scene-0004", "scene-0005", "scene-0006", "scene-0007", "scene-0008", "scene-0009", "scene-0010", "scene-0011", "scene-0019", "scene-0020", "scene-0021", "scene-0022", "scene-0023", "scene-0024", "scene-0025", "scene-0026", "scene-0027", "scene-0028", "scene-0029", "scene-0030", "scene-0031", "scene-0032", "scene-0033", "scene-0034", "scene-0041", "scene-0042", "scene-0043", "scene-0044", "scene-0045", "scene-0046", "scene-0047", "scene-0048", "scene-0049", "scene-0050", "scene-0051", "scene-0052", "scene-0053", "scene-0054", "scene-0055", "scene-0056", "scene-0057", "scene-0058", "scene-0059", "scene-0060", "scene-0061", "scene-0120", "scene-0121", "scene-0122", "scene-0123", "scene-0124", "scene-0125", "scene-0126", "scene-0127", "scene-0128", "scene-0129", "scene-0130", "scene-0131", "scene-0132", "scene-0133", "scene-0134", "scene-0135", "scene-0138", "scene-0139", "scene-0149", "scene-0150", "scene-0151", "scene-0152", "scene-0154", "scene-0155", "scene-0157", "scene-0158", "scene-0159", "scene-0160", "scene-0190", "scene-0191", "scene-0192", "scene-0193", "scene-0194", "scene-0195", "scene-0196", "scene-0347", "scene-0348", "scene-0349", "scene-0350", "scene-0351", "scene-0352", "scene-0353", "scene-0354", "scene-0355", "scene-0356", "scene-0357", "scene-0358", "scene-0359", "scene-0360", "scene-0361", "scene-0362", "scene-0363", "scene-0364", "scene-0365", "scene-0366", "scene-0367", "scene-0368", "scene-0369", "scene-0370", "scene-0371", "scene-0372", "scene-0373", "scene-0374", "scene-0375", "scene-0376", "scene-0377", "scene-0378", "scene-0379", "scene-0380", "scene-0381", "scene-0382", "scene-0383", "scene-0384", "scene-0385", "scene-0386", "scene-0945", "scene-0947", "scene-0949", "scene-0952", "scene-0953", "scene-0955", "scene-0956", "scene-0957", "scene-0958", "scene-0959", "scene-0960", "scene-0961", "scene-0975", "scene-0976", "scene-0977", "scene-0978", "scene-0979", "scene-0980", "scene-0981", "scene-0982", "scene-0983", "scene-0984", "scene-0988", "scene-0989", "scene-0990", "scene-0991", "scene-0177", "scene-0178", "scene-0179", "scene-0180", "scene-0181", "scene-0182", "scene-0183", "scene-0184", "scene-0185", "scene-0187", "scene-0188", "scene-0315", "scene-0316", "scene-0317", "scene-0318", "scene-0420", "scene-0421", "scene-0422", "scene-0423", "scene-0424", "scene-0425", "scene-0426", "scene-0427", "scene-0428", "scene-0429", "scene-0430", "scene-0431", "scene-0432", "scene-0433", "scene-0434", "scene-0435", "scene-0436", "scene-0437", "scene-0438", "scene-0439", "scene-0786", "scene-0787", "scene-0789", "scene-0790", "scene-0791", "scene-0792", "scene-0847", "scene-0848", "scene-0849", "scene-0850", "scene-0851", "scene-0852", "scene-0853", "scene-0854", "scene-0855", "scene-0856", "scene-0858", "scene-0860", "scene-0861", "scene-0862", "scene-0863", "scene-0864", "scene-0865", "scene-0866"]

assert len(train_night) + len(train_day) == 700
assert len(train_rain) + len(train_sunny) == 700

cali_day = set(cali_split_100) & set(train_day) # 88
cali_night = set(cali_split_100) & set(train_night) # 12 
cali_rain = set(cali_split_100) & set(train_rain) # 18
cali_sunny = set(cali_split_100) & set(train_sunny) # 82

cali_day_sunny = set(cali_day) & set(cali_sunny) # 72
cali_day_rain = set(cali_day) & set(cali_rain) # 16
cali_night_sunny = set(cali_night) & set(cali_sunny) # 10
cali_night_rain = set(cali_night) & set(cali_rain) # 2

assert len(cali_night) + len(cali_day) == 100
assert len(cali_rain) + len(cali_sunny) == 100
assert len(set(cali_day_sunny)) + len(set(cali_day_rain)) + len(set(cali_night_sunny)) + len(set(cali_night_rain)) == 100

val_day_ori = \
    ["scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015", "scene-0016", "scene-0017", "scene-0018", "scene-0035", "scene-0036", "scene-0038", "scene-0039", "scene-0092", "scene-0093", "scene-0094", "scene-0095", "scene-0096", "scene-0097", "scene-0098", "scene-0099", "scene-0100", "scene-0101", "scene-0102", "scene-0103", "scene-0104", "scene-0105", "scene-0106", "scene-0107", "scene-0108", "scene-0109", "scene-0110", "scene-0221", "scene-0268", "scene-0269", "scene-0270", "scene-0271", "scene-0272", "scene-0273", "scene-0274", "scene-0275", "scene-0276", "scene-0277", "scene-0278", "scene-0329", "scene-0330", "scene-0331", "scene-0332", "scene-0344", "scene-0345", "scene-0346", "scene-0519", "scene-0520", "scene-0521", "scene-0522", "scene-0523", "scene-0524", "scene-0552", "scene-0553", "scene-0554", "scene-0555", "scene-0556", "scene-0557", "scene-0558", "scene-0559", "scene-0560", "scene-0561", "scene-0562", "scene-0563", "scene-0564", "scene-0565", "scene-0625", "scene-0626", "scene-0627", "scene-0629", "scene-0630", "scene-0632", "scene-0633", "scene-0634", "scene-0635", "scene-0636", "scene-0637", "scene-0638", "scene-0770", "scene-0771", "scene-0775", "scene-0777", "scene-0778", "scene-0780", "scene-0781", "scene-0782", "scene-0783", "scene-0784", "scene-0794", "scene-0795", "scene-0796", "scene-0797", "scene-0798", "scene-0799", "scene-0800", "scene-0802", "scene-0904", "scene-0905", "scene-0906", "scene-0907", "scene-0908", "scene-0909", "scene-0910", "scene-0911", "scene-0912", "scene-0913", "scene-0914", "scene-0915", "scene-0916", "scene-0917", "scene-0919", "scene-0920", "scene-0921", "scene-0922", "scene-0923", "scene-0924", "scene-0925", "scene-0926", "scene-0927", "scene-0928", "scene-0929", "scene-0930", "scene-0931", "scene-0962", "scene-0963", "scene-0966", "scene-0967", "scene-0968", "scene-0969", "scene-0971", "scene-0972"] 
val_night_ori = \
    ['scene-1059', 'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067', 'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
val_sunny_ori = \
    ["scene-0003", "scene-0012", "scene-0013", "scene-0014", "scene-0015", "scene-0016", "scene-0017", "scene-0018", "scene-0035", "scene-0036", "scene-0038", "scene-0039", "scene-0092", "scene-0093", "scene-0094", "scene-0095", "scene-0096", "scene-0097", "scene-0098", "scene-0099", "scene-0100", "scene-0101", "scene-0102", "scene-0103", "scene-0104", "scene-0105", "scene-0106", "scene-0107", "scene-0108", "scene-0109", "scene-0110", "scene-0221", "scene-0268", "scene-0269", "scene-0270", "scene-0271", "scene-0272", "scene-0273", "scene-0274", "scene-0275", "scene-0276", "scene-0277", "scene-0278", "scene-0329", "scene-0330", "scene-0331", "scene-0332", "scene-0344", "scene-0345", "scene-0346", "scene-0519", "scene-0520", "scene-0521", "scene-0522", "scene-0523", "scene-0524", "scene-0552", "scene-0553", "scene-0554", "scene-0555", "scene-0556", "scene-0557", "scene-0558", "scene-0559", "scene-0560", "scene-0561", "scene-0562", "scene-0563", "scene-0564", "scene-0565", "scene-0770", "scene-0771", "scene-0775", "scene-0777", "scene-0778", "scene-0780", "scene-0781", "scene-0782", "scene-0783", "scene-0784", "scene-0794", "scene-0795", "scene-0796", "scene-0797", "scene-0798", "scene-0799", "scene-0800", "scene-0802", "scene-0916", "scene-0917", "scene-0919", "scene-0920", "scene-0921", "scene-0922", "scene-0923", "scene-0924", "scene-0925", "scene-0926", "scene-0927", "scene-0928", "scene-0929", "scene-0930", "scene-0931", "scene-0962", "scene-0963", "scene-0966", "scene-0967", "scene-0968", "scene-0969", "scene-0971", "scene-0972", "scene-1059", "scene-1061", "scene-1062", "scene-1063", "scene-1064", "scene-1066", "scene-1068", "scene-1069", "scene-1070", "scene-1071", "scene-1072", "scene-1073"]     
val_rain_ori = \
    ["scene-0625", "scene-0626", "scene-0627", "scene-0629", "scene-0630", "scene-0632", "scene-0633", "scene-0634", "scene-0635", "scene-0636", "scene-0637", "scene-0638", "scene-0904", "scene-0905", "scene-0906", "scene-0907", "scene-0908", "scene-0909", "scene-0910", "scene-0911", "scene-0912", "scene-0913", "scene-0914", "scene-0915", "scene-1060", "scene-1065", "scene-1067"]
val_usa = \
    ["scene-0909", "scene-0910", "scene-0911", "scene-0912", "scene-0913", "scene-0914", "scene-0915"] 


val_daysunny = list(set(val_day_ori) & set(val_sunny_ori))
val_dayrain = list(set(val_day_ori) & set(val_rain_ori))
val_nightsunny = list(set(val_night_ori) & set(val_sunny_ori))
val_nightrain = list(set(val_night_ori) & set(val_rain_ori))


# f_scores = open("states_scores_3donly_2.log", "w")
f_calib_ncscore_files = [open(f"/home/minkycho/SparseFusion/calibration/calib_ncscore_analysis_cls{i}.log", "w") for i in range(10)]

sample_id = 0


NAME_ABREV = {
    "car": "C",
    "truck": "TR",
    "construction_vehicle": "CV",
    "bus": "B",
    "trailer": "TL",
    "barrier": "BR",
    "motorcycle": "M",
    "bicycle": "BC",
    "pedestrian": "P",
    "traffic_cone": "TC",
}


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


# Initialize the NuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/y/minkycho/nuscenes_empty_conformalcal100/', verbose=True)

def visualize_bbox(data=None, data_name=None, outputs=None, cfg=None, mode="pred", topk_ncscore_dict=None, topk_p_val_dict=None, topk_top3_values_dict=None, topk_top3_indices_dict=None):
    global sample_id
    out_dir = "/adafuse/viz_results/smallset_val/viz_lidaronly_alphabeta_tmp2"
    fusion_mode = "adaptive"
    mode = "pred"


    # print("=" * 50)
    # print(
    #     f"[!CHECK! @ visualize_results.py] output directory for visualized results: {out_dir}\n"
    #     f"[!CHECK! @ visualize_results.py] current visualization mode: {mode}"
    # )
    # print("=" * 50)

    metas = data["img_metas"][0].data[0][0]

    bbox_classes = None
    bbox_score = 0.3
    map_score = 0.5

    if mode == "gt" and "gt_bboxes_3d" in data:
        bboxes = data["gt_bboxes_3d"][0].data[0][0].tensor.numpy() # MK: scores?
        labels = data["gt_labels_3d"][0].data[0][0].numpy()
        assert data["gt_labels_3d"][0].data[0][0].numpy().shape[0] == data["gt_bboxes_3d"][0].data[0][0].tensor.numpy().shape[0]
        
        scores = None
        topk_ncscore_dict = None
        topk_p_val_dict = None

        if bbox_classes is not None:
            indices = np.isin(labels, bbox_classes)
            bboxes = bboxes[indices]
            labels = labels[indices]

        # bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        
        
    elif mode == "pred" and "boxes_3d" in outputs:
        bboxes = outputs["boxes_3d"].tensor.numpy()
        scores = outputs["scores_3d"].numpy()
        labels = outputs["labels_3d"].numpy()
        

        for i in range(6):
            topk_ncscore_dict[f"img{i}"] = topk_ncscore_dict[f"img{i}"].numpy()
            topk_ncscore_dict[f"pts{i}"] = topk_ncscore_dict[f"pts{i}"].numpy()
            topk_p_val_dict[f"img{i}"] = topk_p_val_dict[f"img{i}"].numpy()
            topk_p_val_dict[f"pts{i}"] = topk_p_val_dict[f"pts{i}"].numpy()

            topk_top3_values_dict[f"img{i}"] = topk_top3_values_dict[f"img{i}"].numpy()
            topk_top3_values_dict[f"pts{i}"] = topk_top3_values_dict[f"pts{i}"].numpy()

            topk_top3_indices_dict[f"img{i}"] = topk_top3_indices_dict[f"img{i}"].numpy()
            topk_top3_indices_dict[f"pts{i}"] = topk_top3_indices_dict[f"pts{i}"].numpy()
            
            
        if bbox_classes is not None:
            indices = np.isin(labels, bbox_classes)
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            for i in range(6):
                topk_ncscore_dict[f"img{i}"] = topk_ncscore_dict[f"img{i}"][indices]
                topk_ncscore_dict[f"pts{i}"] = topk_ncscore_dict[f"pts{i}"][indices]
                topk_p_val_dict[f"img{i}"] = topk_p_val_dict[f"img{i}"][indices]
                topk_p_val_dict[f"pts{i}"] = topk_p_val_dict[f"pts{i}"][indices]
                
                topk_top3_values_dict[f"img{i}"] = topk_top3_values_dict[f"img{i}"][indices]
                topk_top3_values_dict[f"pts{i}"] = topk_top3_values_dict[f"pts{i}"][indices]

                topk_top3_indices_dict[f"img{i}"] = topk_top3_indices_dict[f"img{i}"][indices]
                topk_top3_indices_dict[f"pts{i}"] = topk_top3_indices_dict[f"pts{i}"][indices]


        if bbox_score is not None:
            indices = scores >= bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]
            for i in range(6):
                topk_ncscore_dict[f"img{i}"] = topk_ncscore_dict[f"img{i}"][indices]
                topk_ncscore_dict[f"pts{i}"] = topk_ncscore_dict[f"pts{i}"][indices]
                topk_p_val_dict[f"img{i}"] = topk_p_val_dict[f"img{i}"][indices]
                topk_p_val_dict[f"pts{i}"] = topk_p_val_dict[f"pts{i}"][indices]
                topk_top3_values_dict[f"img{i}"] = topk_top3_values_dict[f"img{i}"][indices]
                topk_top3_values_dict[f"pts{i}"] = topk_top3_values_dict[f"pts{i}"][indices]

                topk_top3_indices_dict[f"img{i}"] = topk_top3_indices_dict[f"img{i}"][indices]
                topk_top3_indices_dict[f"pts{i}"] = topk_top3_indices_dict[f"pts{i}"][indices]


            # print(
            #     f"indices: {indices} topk_ncscore_dict: {topk_ncscore_dict} topk_p_val_dict: {topk_p_val_dict}"
            # )
            # f_scores.write(
            #     f"length: {len(scores)} sum: {sum(scores)} mean: {sum(scores)/len(scores)} min: {min(scores)} max: {max(scores)}\n"
            # )

        for i in range(6):
            assert len(scores) == len(topk_ncscore_dict[f"img{i}"])
            assert len(scores) == len(topk_ncscore_dict[f"pts{i}"])
            assert len(scores) == len(topk_p_val_dict[f"img{i}"])
            assert len(scores) == len(topk_p_val_dict[f"pts{i}"])
        
        """
        * Calibration Data 
        1. get scene number (from sample token) and get which directory it belongs to (DS/DR/NS/NR) -> V
        For each existing object, 
            2. compute distance and categoriza this into near, mid, far
            3. compute the object size and categorize this into small, med, large
            4. print out class, distance_val, distance_cat, size_val, size_cat, DS/DR/NS/NR, NC Score (:.6f)
        """   

        gt_bboxes = data["gt_bboxes_3d"][0].data[0][0].tensor.numpy()
        gt_labels = data["gt_labels_3d"][0].data[0][0].numpy()
        gt_taken = set()

        for pred_idx, pred_box in enumerate(bboxes):
            min_dist = np.inf
            match_gt_idx = None
            match_gt_box = None

            for gt_idx, gt_box in enumerate(gt_bboxes):
                if gt_labels[gt_idx] == labels[pred_idx] and not gt_idx in gt_taken:
                    this_distance = np.linalg.norm(np.array(pred_box[:2]) - np.array(gt_box[:2]))
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx
                        match_gt_box = gt_box
            is_match = min_dist < 4.0
            
            if not is_match:
                continue
                
            sample_token = metas['sample_idx']
            scene_token = nusc.get('sample', sample_token)['scene_token']
            scene = nusc.get('scene', scene_token)
            scene_name = scene['name']
            
            # sample_token = None
            # scene_token = None
            # scene = None
            # scene_name = None

            driving_condition = None # DS, DR, NS, NR
            if scene_name in val_daysunny: #cali_day_sunny:
                driving_condition = "DS"
            elif scene_name in val_dayrain: #cali_day_rain:
                driving_condition = "DR"
            elif scene_name in val_nightsunny: #cali_night_sunny:
                driving_condition = "NS"
            elif scene_name in val_nightrain: #cali_night_rain:
                driving_condition = "NR"
            else:
                driving_condition = "NONE"
                
            dist_ego = np.sqrt(np.sum(match_gt_box[0:2] ** 2)) 
            # Categorize according to dist_ego (near, mid, far) 
            # if dist < 20, near
            # if 20 <= dist < 30, mid1
            # if 30 <= dist < 40, mid2
            # if 40 <= dist, far
            dist_cat = None
            if dist_ego < 20:
                dist_cat = "near"
            elif dist_ego < 40:
                dist_cat = "mid"
            else:
                dist_cat = "far"  
                    
            obj_size = max(match_gt_box[3], match_gt_box[4], match_gt_box[5])
            # obj_size = np.sqrt(match_gt_box[3] ** 2 + match_gt_box[4] ** 2 + match_gt_box[5] ** 2) 
            # Categorize according to obj_size (small, med, big) 
            # if size < 3, small
            # if 3 <= size < 5, med1
            # if 5 <= size < 7, med2
            # if 7 <= size < 9, big1
            # if 9 <= size, big2
            obj_size_cat = None
            if obj_size < 2:
                obj_size_cat = "small"
            elif obj_size < 4:
                obj_size_cat = "med"
            else:
                obj_size_cat = "big"
                
        
            f_calib_ncscore_files[labels[pred_idx]].write(f"{cfg.class_names[labels[pred_idx]]} {scores[pred_idx]*100:.2f} {dist_ego:.4f} {dist_cat} {obj_size:.4f} {obj_size_cat} {driving_condition} Layer0 {topk_nc_dict['img0'][pred_idx]:.6f} {topk_nc_dict['pts0'][pred_idx]:.6f} {scene_name} {scene_token} {sample_token}\n")
                
        # bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

    else:
        bboxes = None
        labels = None
    
