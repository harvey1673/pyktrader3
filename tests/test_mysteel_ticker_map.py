import unittest

from tests.mysteel_ticker_map import mysteel_ticker_map, ticker_name


EXPECTED_CODES = {
    'ID00184174',
    'ID00184175',
    'ID00184176',
    'ID00184178',
    'ID00184490',
    'ID00184492',
    'ID00184494',
    'ID00184497',
    'ID00186001',
    'ID00186597',
    'ID00187013',
    'ID00187443',
    'ID00187978',
    'ID00188061',
    'ID00188062',
    'ID00188063',
    'ID00188064',
    'ID00188065',
    'ID00188307',
    'ID00188314',
    'ID00188315',
    'ID00188359',
    'ID00375008',
    'ID00384633',
    'ID00408152',
    'ID00408153',
    'ID00408155',
    'ID01002072',
    'ID01002311',
    'ID01011788',
    'ID01017688',
    'ID01027073',
    'ID01030576',
    'ID01037437',
    'ID01109378',
    'ID01167269',
    'ID01167270',
    'ID01167591',
    'ID01167594',
    'ID01168763',
    'ID01200757',
    'ID01201815',
    'ID01207170',
    'ID01214595',
    'ID01216483',
    'ID01218647',
    'ID01230664',
    'ID01232909',
    'ID01245758',
    'ID01301727',
    'ID01301728',
    'ID01301853',
    'ID01349545',
    'ID01369403',
    'ID01370598',
    'ID01490913',
    'ID01508544',
    'ID01517441',
    'ID01532024',
    'ID01709994',
    'ID01718576',
    'ID01718581',
    'ID01718582',
    'ID01719869',
    'ID01720310',
    'ID01721655',
    'ID01721691',
    'ID01721692',
    'ID01721697',
    'ID01733105',
    'ID01736015',
    'ID01835359',
    'ID01835361',
    'ID01862250',
    'ID01881060',
    'ID01891248',
    'ID01897968',
    'ID01990129',
    'ID01998696',
    'ID02026458',
    'ID02032074',
    'ID02069031',
    'ID02069937',
    'ID02215004',
    'ID02343778',
    'ID02424817',
    'RE00010184',
    'RE00010776',
    'RE00024787',
    'RE00024794',
    'RE00024799',
    'RE00024806',
    'RE00033240',
}


class MysteelTickerMapTests(unittest.TestCase):
    def test_contains_every_workbook_ticker(self):
        self.assertEqual(set(ticker_name), EXPECTED_CODES)
        self.assertEqual(len(ticker_name), 93)

    def test_aliases_are_unique_and_nonempty(self):
        aliases = list(ticker_name.values())

        self.assertTrue(all(alias.strip() for alias in aliases))
        self.assertEqual(len(aliases), len(set(aliases)))

    def test_uses_same_mapping_for_source_qualified_alias(self):
        self.assertIs(mysteel_ticker_map, ticker_name)

    def test_representative_names_match_normalized_headers(self):
        self.assertEqual(ticker_name['ID01718576'], '精煤_523家样本矿山_库存周')
        self.assertEqual(ticker_name['ID02026458'], '多晶硅_n型_致密料_生产成本周')
        self.assertEqual(ticker_name['RE00024806'], '煤炭_进口_蒙古产_通关量_甘其毛都口岸日')


if __name__ == '__main__':
    unittest.main()
