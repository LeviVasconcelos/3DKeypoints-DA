import xml.etree.ElementTree as ET
import numpy as np

def _parse_matrix(string):
      rows_split = string[1:-1].split(";")
      numbers = np.asarray([p.split(' ') for p in rows_split]).ravel()
      array = np.asarray([int(x) for x in numbers])
      return array.reshape(-1, 2)

class H36M_Metadata:
    def __init__(self, metadata_file):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []
        self.camera_resolutions = {}
        subject_from_xml = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
        tree = ET.parse(metadata_file)
        root = tree.getroot()
        self.root_ = root
        self.camera_ids = [elem.text for elem in root.find('dbcameras/index2id')]
        for i, tr in enumerate(root.find('mapping')):
            td_list = [td.text for td in tr]
            a,b, args = td_list[0], td_list[1], td_list[2:] 
            if i == 0:
                subjects_list = [td.text for td in tr]
                self.subjects = subjects_list[2:]
                self.sequence_mappings = {subject: {} for subject in self.subjects}
            elif i < 33:
                prefix_list = [td.text for td in tr]
                action_id, subaction_id, prefixes = prefix_list[0], prefix_list[1], prefix_list[2:]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)] = prefix
            if a == b == None:
                  for i,e in enumerate(args):
                        res_matrix = _parse_matrix(e)
                        for k,res in enumerate(res_matrix):
                              self.sequence_mappings[subject_from_xml[i]][(self.camera_ids[k], '')] = res

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        

    def get_base_filename(self, subject, action, subaction, camera):
        return '{}.{}'.format(self.sequence_mappings[subject][(action, subaction)], camera)


def load_h36m_metadata():
    return H36M_Metadata('metadata.xml')


if __name__ == '__main__':
    metadata = load_h36m_metadata()
    print(metadata.subjects)
    print(metadata.sequence_mappings)
    print(metadata.action_names)
    print(metadata.camera_ids)
