#!/usr/bin/python
# -*- coding: utf-8 -*-

import utils

class TDABCHyperParamFileParser:
    def __init__(self):
        self.values_dict = {}

    def process_hp_file(self, file_name):
        hpfile = open(file_name, "r")

        for line in hpfile.readlines():

            if len(line) < 10:                                              # for jump the white lines
                continue

            if line.find("self.") != -1 or \
                line.find("CROSS") != -1 or \
                line.find("RIPS") != -1:
                continue

            parts = line.split(",")                                         # memory and timing files has the same structure
            nkey = ""
            values = {}
            for part in parts:
                sub_parts = part.split("=")
                num_name = sub_parts[0].strip()
                float_num = sub_parts[-1].strip()
                values.update({num_name: float_num})
                if len(num_name) == 1:
                    nkey = "{0}{1}".format(nkey, float_num)
                elif len(num_name) > 1:
                    values[num_name] = [float_num]
            can_i_update = True
            if nkey in self.values_dict:
                for e in self.values_dict[nkey]:
                    if len(e) > 1 and e in values:
                        self.values_dict[nkey][e].extend(values[e])
                        can_i_update = False
            if can_i_update:
                self.values_dict.update({nkey: values})

        hpfile.close()

        return self.values_dict

    def get_hyper_params(self):
        files_names = utils.get_all_filenames()

        for file in files_names:
            self.process_hp_file(file)

        return self.values_dict

    def get_float_value(self, value2check):
        if (value2check is None):
            return None

        if (value2check == "None"):
            return None

        if (type(value2check) == list):
            if (value2check[0] is None or value2check[0] == "None"):
                return None
            return float(value2check[0])

        return float(value2check)

    def get_avg(self, values_list):
        sum = 0
        for i in values_list:
            sum += i
        size = len(values_list)
        if size > 0:
            return sum / size

        return None

    def merge_all_data(self):

        if not self.values_dict:
            self.get_hyper_params()

        all_data = dict()

        for key in self.values_dict:
            value = self.values_dict[key]

            r = value["r"]
            q2 = value["q"]

            timing = None
            mem_bytes = None
            mem_kb = None
            mem_mb = None
            mem_gb = None
            memory_size = None

            if ("timing" in value):
                timing = self.get_float_value(value["timing"])
            if ("mem_bytes" in value):
                mem_bytes = self.get_float_value(value["mem_bytes"])
            if ("mem_kb" in value):
                mem_kb = self.get_float_value(value["mem_kb"])
            if ("mem_mb" in value):
                mem_mb = self.get_float_value(value["mem_mb"])
            if ("mem_gb" in value):
                mem_gb = self.get_float_value(value["mem_gb"])
            if ("memory_size" in value):
                memory_size = self.get_float_value(value["memory_size"])

            if not r in all_data:
                all_data.update({r: {}})
            akqr = all_data[r]

            if not q2 in akqr:
                akqr.update({q2: {}})

            akqrq = akqr[q2]
            if not ("timing" in akqrq):
                akqrq.update({"timing": []})
            if not ("mem_bytes" in akqrq):
                akqrq.update({"mem_bytes": []})
            if not ("mem_kb" in akqrq):
                akqrq.update({"mem_kb": []})
            if not ("mem_mb" in akqrq):
                akqrq.update({"mem_mb": []})
            if not ("mem_gb" in akqrq):
                akqrq.update({"mem_gb": []})
            if not ("memory_size" in akqrq):
                akqrq.update({"memory_size": []})

            if not (timing is None):
                akqrq["timing"].append(timing)

            if not (mem_bytes is None):
                akqrq["mem_bytes"].append(mem_bytes)

            if not (mem_kb is None):
                akqrq["mem_kb"].append(mem_kb)

            if not (mem_mb is None):
                akqrq["mem_mb"].append(mem_mb)

            if not (mem_gb is None):
                akqrq["mem_gb"].append(mem_gb)

            if not (memory_size is None):
                akqrq["memory_size"].append(memory_size)

        for r in all_data:
            for q in all_data:
                value = all_data[r][q]
                if ("timing" in value):
                    timing_list = value["timing"]
                    value["timing"] = self.get_avg(timing_list)
                if ("mem_bytes" in value):
                    mem_byte_list = value["mem_bytes"]
                    value["mem_bytes"] = self.get_avg(mem_byte_list)
                if ("mem_kb" in value):
                    mem_kb_list = value["mem_kb"]
                    value["mem_kb"] = self.get_avg(mem_byte_list)
                if ("mem_mb" in value):
                    mem_mb_list = value["mem_mb"]
                    value["mem_mb"] = self.get_avg(mem_mb_list)
                if ("mem_gb" in value):
                    mem_gb_list = value["mem_gb"]
                    value["mem_gb"] = self.get_avg(mem_gb_list)
                if ("memory_size" in value):
                    memory_size_list = value["memory_size"]
                    value["memory_size"] = self.get_avg(memory_size_list)

        return all_data

