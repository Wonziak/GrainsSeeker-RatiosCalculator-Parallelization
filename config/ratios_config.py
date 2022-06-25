ratiosToCalculateList = ['Malinowska',
                         'Blair Bliss',
                         'Danielsson',
                         'Haralick',
                         'Mz',
                         'RLS',
                         'RF',
                         'RC1',
                         'RC2',
                         'RCOM',
                         'LP1',
                         'LP2',
                         'LP3']


def to_lower_case():
    ratiosToCalculateList[:] = [item.lower() for item in ratiosToCalculateList]
