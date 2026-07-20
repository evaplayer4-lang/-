# -*- coding: utf-8 -*-
import arcpy
import csv
import os
import sys
#reload(sys)
sys.setdefaultencoding('utf8')

CSV_FILE = r"E:\Python1\beijing_population.csv"
SHP_NAME_FIELD = "name"

def read_csv_data():
    """读取制表符分隔的CSV，返回 {地名: [所属区, pop2000, pop2010, pop2020]} """
    data_dict = {}
    current_district = ""
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f, delimiter="\t")   # 关键修正：制表符
        header = next(reader)
        # 跳过可能的BOM
        if header[0].startswith('\ufeff'):
            header[0] = header[0].replace('\ufeff', '')
        for row in reader:
            if len(row) < 5:
                continue
            name = row[0].strip()
            type_ = row[1].strip()
            pop2000 = int(row[2]) if row[2] else 0
            pop2010 = int(row[3]) if row[3] else 0
            pop2020 = int(row[4]) if row[4] else 0

            if type_ == u"\u533a":   # 区
                current_district = name
            elif type_ in (u"\u8857\u9053", u"\u9547", u"\u4e61"):  # 街道,镇,乡
                data_dict[name] = [current_district, pop2000, pop2010, pop2020]
    return data_dict

def get_xy(geom):
    """从质心几何对象或坐标元组中提取(x,y)"""
    if isinstance(geom, tuple):
        return geom
    try:
        pnt = geom.centroid
        return (pnt.X, pnt.Y)
    except:
        try:
            return (geom.X, geom.Y)
        except:
            raise Exception("Cannot extract coordinates")

if __name__ == "__main__":
    try:
        arcpy.env.overwriteOutput = True
        input_shp = arcpy.GetParameterAsText(0)
        out_folder = arcpy.GetParameterAsText(1)
        out_name = arcpy.GetParameterAsText(2)
        out_shp = os.path.join(out_folder, out_name + ".shp")

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        sr = arcpy.Describe(input_shp).spatialReference
        arcpy.CreateFeatureclass_management(out_folder, out_name, "POINT", "", "", "", sr)

        arcpy.AddField_management(out_shp, "ST_NAME", "TEXT", field_length=50)
        arcpy.AddField_management(out_shp, "COUNTY", "TEXT", field_length=30)
        arcpy.AddField_management(out_shp, "POP2000", "LONG")
        arcpy.AddField_management(out_shp, "POP2010", "LONG")
        arcpy.AddField_management(out_shp, "POP2020", "LONG")

        town_data = read_csv_data()
        arcpy.AddMessage(u"CSV loaded: " + str(len(town_data)) + u" records")

        match = 0
        unmatch = 0

        with arcpy.da.SearchCursor(input_shp, [SHP_NAME_FIELD, "SHAPE@TRUECENTROID"]) as s_cur:
            with arcpy.da.InsertCursor(out_shp, ["SHAPE@XY", "ST_NAME", "COUNTY", "POP2000", "POP2010", "POP2020"]) as i_cur:
                for name, raw_geom in s_cur:
                    shp_name = str(name).strip()
                    xy = get_xy(raw_geom)
                    county, p00, p10, p20 = u"\u65e0\u5339\u914d", 0, 0, 0

                    if shp_name in town_data:
                        county, p00, p10, p20 = town_data[shp_name]
                        match += 1
                    else:
                        short = shp_name.replace(u"\u8857\u9053", "").replace(u"\u9547", "").replace(u"\u4e61", "").strip()
                        found = False
                        for csv_name in town_data:
                            csv_short = csv_name.replace(u"\u8857\u9053", "").replace(u"\u9547", "").replace(u"\u4e61", "").strip()
                            if csv_short == short:
                                county, p00, p10, p20 = town_data[csv_name]
                                match += 1
                                found = True
                                break
                        if not found:
                            unmatch += 1

                    i_cur.insertRow([xy, shp_name, county, p00, p10, p20])

        arcpy.AddMessage(u"\u5339\u914d\u6210\u529f: " + str(match) + u" \u4e2a")
        arcpy.AddMessage(u"\u672a\u5339\u914d: " + str(unmatch) + u" \u4e2a")
        arcpy.AddMessage(u"\u4efb\u52a1\u5b8c\u6210! \u6587\u4ef6: " + out_shp)

    except Exception as e:
        arcpy.AddError(u"\u9519\u8bef: " + str(e))