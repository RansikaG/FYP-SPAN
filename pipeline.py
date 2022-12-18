from CPDM import CPDM

areas = CPDM()
areas.get_area_ratios(image_name="Acura_ILX_2019_25_17_200_24_4_70_55_182_24_FWD_5_4_4dr_rMu.jpg")
areas.cooccurence_attention(image_1="Acura_ILX_2019_25_17_200_24_4_70_55_182_24_FWD_5_4_4dr_rMu.jpg",
                            image_2="Acura_MDX_2019_44_18_290_35_6_77_67_196_20_FWD_7_4_SUV_ueu.jpg")
