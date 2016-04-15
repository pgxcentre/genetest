

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import unittest
from tempfile import TemporaryDirectory
from itertools import zip_longest as zip

import numpy as np
import pandas as pd
from pyplink import PyPlink

from ..genotypes.core import GenotypesContainer, Representation, \
                             MarkerGenotypes
from ..genotypes.plink import PlinkGenotypes
from ..genotypes.impute2 import Impute2Genotypes


__copyright__ = "Copyright 2016, Beaulieu-Saucier Pharmacogenomics Centre"
__license__ = "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)"


class TestCore(unittest.TestCase):
    def test_get_genotypes(self):
        """Tests that 'get_genotypes' raises a NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            GenotypesContainer().get_genotypes("marker")

    def test_iter_marker_genotypes(self):
        """Tests that 'iter_marker_genotypes' raises a NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            GenotypesContainer().iter_marker_genotypes()

    def test_close(self):
        """Tests that 'close' raises a NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            GenotypesContainer().close()

    def test_check_valid_representation(self):
        """Tests with a valid representation."""
        GenotypesContainer.check_representation(Representation.ADDITIVE)

    def test_check_invalid_representation(self):
        """Tests with an invalid representation."""
        with self.assertRaises(ValueError) as cm:
            GenotypesContainer.check_representation("invalid_representation")
        self.assertEqual(
            "INVALID_REPRESENTATION is an invalid representation",
            str(cm.exception),
        )

    def test_additive2genotypic(self):
        """Tests the 'additive2genotypic' function."""
        data = pd.DataFrame(
            [("sample_0", 0, 0, 0),
             ("sample_1", 0, 0, 0),
             ("sample_2", 1, 1, 0),
             ("sample_3", 0, 0, 0),
             ("sample_4", 1, 1, 0),
             ("sample_5", 0, 0, 0),
             ("sample_6", np.nan, np.nan, np.nan),
             ("sample_7", 2, 0, 1),
             ("sample_8", 1, 1, 0),
             ("sample_9", 2, 0, 1)],
            columns=["iid", "geno", "geno_ab", "geno_bb"],
        ).set_index("iid")

        observed = GenotypesContainer.additive2genotypic(data.loc[:, ["geno"]])

        self.assertTrue(
            data.loc[:, ["geno_ab", "geno_bb"]].equals(observed)
        )

    def test_dosage2additive(self):
        """Tests the 'dosage2additive' function."""
        data = pd.DataFrame(
            [("sample_0", 0.0, 0.0),
             ("sample_1", 0.1, 0.0),
             ("sample_2", 0.9, 1.0),
             ("sample_3", 1.0, 1.0),
             ("sample_4", 1.1, 1.0),
             ("sample_5", 1.9, 2.0),
             ("sample_6", 2.0, 2.0),
             ("sample_7", 2.1, 2.0),
             ("sample_8", np.nan, np.nan),
             ("sample_9", 1.3, 1.0)],
            columns=["iid", "geno", "geno_add"],
        ).set_index("iid")

        observed = GenotypesContainer.dosage2additive(data.loc[:, ["geno"]])

        expected = data.loc[:, ["geno_add"]].rename(
            columns={"geno_add": "geno"},
        )
        self.assertTrue(
            expected.equals(observed)
        )

    def test_create_geno_df(self):
        """Tests the 'create_geno_df' function."""
        expected = pd.DataFrame(
            [("sample_1", 0),
             ("sample_2", 1),
             ("sample_3", np.nan),
             ("sample_4", 2)],
            columns=["sample", "geno"],
        ).set_index("sample")

        observed = PlinkGenotypes.create_geno_df(
            genotypes=[0, 1, -1, 2],
            samples=["sample_{}".format(i+1) for i in range(4)],
        )

        self.assertTrue(expected.equals(observed))

    def test_check_genotypes_no_flip(self):
        """Tests the 'check_genotypes' function (no flip)."""
        geno = pd.DataFrame({"geno": [0, 0, 0, np.nan, 1, 1, 2, 2]})
        obs_geno, obs_minor, obs_major = PlinkGenotypes.check_genotypes(
            genotypes=geno,
            minor="B",
            major="A",
        )

        self.assertTrue(geno.equals(obs_geno))
        self.assertEqual("B", obs_minor)
        self.assertEqual("A", obs_major)

    def test_check_genotypes_flip(self):
        """Tests the 'check_genotypes' function (flip)."""
        geno = pd.DataFrame({"geno": [0, np.nan, np.nan, np.nan, 1, 2, 2, 2]})
        obs_geno, obs_minor, obs_major = PlinkGenotypes.check_genotypes(
            genotypes=geno,
            minor="B",
            major="A",
        )

        geno = pd.DataFrame({"geno": [2, np.nan, np.nan, np.nan, 1, 0, 0, 0]})
        self.assertTrue(geno.equals(obs_geno))
        self.assertEqual("A", obs_minor)
        self.assertEqual("B", obs_major)


class TestPlink(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory(prefix="project_x_")

        # Creating a BED file
        self.prefix = os.path.join(self.tmpdir.name, "test_plink_file")
        with PyPlink(self.prefix, "w") as bed:
            bed.write_genotypes([0, 0, 0, 1, 0, 0, -1, 2, 1, 0])
            bed.write_genotypes([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
            bed.write_genotypes([2, 0, 2, 2, 1, 1, -1, 2, 2, 1])
            bed.write_genotypes([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

        # Creating a BIM file
        with open(self.prefix + ".bim", "w") as bim:
            print(1, "marker_1", 0, 1, "C", "T", sep="\t", file=bim)
            print(1, "marker_2", 0, 2, "G", "C", sep="\t", file=bim)
            print(2, "marker_3", 0, 100, "A", "T", sep="\t", file=bim)
            print(3, "marker_4", 0, 230, "G", "T", sep="\t", file=bim)

        # Creating a FAM file
        with open(self.prefix + ".fam", "w") as fam:
            print("f0", "i0", 0, 0, 1, -9, file=fam)
            print("f1", "i1", 0, 0, 1, -9, file=fam)
            print("f2", "i2", 0, 0, 2, -9, file=fam)
            print("f3", "i3", 0, 0, 2, -9, file=fam)
            print("f4", "i4", 0, 0, 1, -9, file=fam)
            print("f5", "i5", 0, 0, 2, -9, file=fam)
            print("f6", "i6", 0, 0, 1, -9, file=fam)
            print("f7", "i7", 0, 0, 1, -9, file=fam)
            print("f8", "i8", 0, 0, 1, -9, file=fam)
            print("f9", "i9", 0, 0, 2, -9, file=fam)

        # The expected results (additive)
        self.expected_marker_1_add = pd.DataFrame(
                [("i0", 0), ("i1", 0), ("i2", 0), ("i3", 1), ("i4", 0),
                 ("i5", 0), ("i6", np.nan), ("i7", 2), ("i8", 1), ("i9", 0)],
                columns=["iid", "geno"],
            ).set_index("iid")
        self.expected_marker_2_add = pd.DataFrame(
                [("i0", 0.0), ("i1", 0.0), ("i2", 0.0), ("i3", 0.0),
                 ("i4", 1.0), ("i5", 1.0), ("i6", 0.0), ("i7", 0.0),
                 ("i8", 0.0), ("i9", 1.0)],
                columns=["iid", "geno"],
            ).set_index("iid")
        self.expected_marker_3_add = pd.DataFrame(
                [("i0", 0), ("i1", 2), ("i2", 0), ("i3", 0), ("i4", 1),
                 ("i5", 1), ("i6", np.nan), ("i7", 0), ("i8", 0), ("i9", 1)],
                columns=["iid", "geno"],
            ).set_index("iid")
        self.expected_marker_4_add = pd.DataFrame(
                [("i0", np.nan), ("i1", np.nan), ("i2", np.nan),
                 ("i3", np.nan), ("i4", np.nan), ("i5", np.nan), ("i6",
                 np.nan), ("i7", np.nan), ("i8", np.nan), ("i9", np.nan)],
                columns=["iid", "geno"],
            ).set_index("iid")

        # The expected results (genotypic)
        self.expected_marker_1_geno = pd.DataFrame(
                [("i0", 0, 0), ("i1", 0, 0), ("i2", 0, 0), ("i3", 1, 0),
                 ("i4", 0, 0), ("i5", 0, 0), ("i6", np.nan, np.nan),
                 ("i7", 0, 1), ("i8", 1, 0), ("i9", 0, 0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        self.expected_marker_2_geno = pd.DataFrame(
                [("i0", 0.0, 0.0), ("i1", 0.0, 0.0), ("i2", 0.0, 0.0),
                 ("i3", 0.0, 0.0), ("i4", 1.0, 0.0), ("i5", 1.0, 0.0),
                 ("i6", 0.0, 0.0), ("i7", 0.0, 0.0), ("i8", 0.0, 0.0),
                 ("i9", 1.0, 0.0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        self.expected_marker_3_geno = pd.DataFrame(
                [("i0", 0, 0), ("i1", 0, 1), ("i2", 0, 0), ("i3", 0, 0),
                 ("i4", 1, 0), ("i5", 1, 0), ("i6", np.nan, np.nan),
                 ("i7", 0, 0), ("i8", 0, 0), ("i9", 1, 0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        self.expected_marker_4_geno = pd.DataFrame(
                [("i0", np.nan, np.nan), ("i1", np.nan, np.nan),
                 ("i2", np.nan, np.nan), ("i3", np.nan, np.nan),
                 ("i4", np.nan, np.nan), ("i5", np.nan, np.nan),
                 ("i6", np.nan, np.nan), ("i7", np.nan, np.nan),
                 ("i8", np.nan, np.nan), ("i9", np.nan, np.nan)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init(self):
        """Tests the creation of a PlinkGenotypes instance."""
        observed = PlinkGenotypes(self.prefix)

        # We don't check the BED and the BIM file, since they come from
        # PyPlink. We only check the FAM file, because it might be modified.
        self.assertEqual(
            ["i{}".format(i) for i in range(10)],
            list(observed.fam.index.values),
        )

    def test_init_duplicated_iid(self):
        """Tests the creation of the instance, but with duplicated IID."""
        # Modifying the FAM file
        with open(self.prefix + ".fam", "w") as fam:
            print("f0", "i0", 0, 0, 1, -9, file=fam)
            print("f1", "i1", 0, 0, 1, -9, file=fam)
            print("f2", "i2", 0, 0, 2, -9, file=fam)
            print("f3", "i3", 0, 0, 2, -9, file=fam)
            print("f4", "i4", 0, 0, 1, -9, file=fam)
            print("f5", "i0", 0, 0, 2, -9, file=fam)
            print("f6", "i6", 0, 0, 1, -9, file=fam)
            print("f7", "i7", 0, 0, 1, -9, file=fam)
            print("f8", "i8", 0, 0, 1, -9, file=fam)
            print("f9", "i9", 0, 0, 2, -9, file=fam)

        observed = PlinkGenotypes(self.prefix)

        # The FAM index should now be fid_iid
        expected = ["f{i}_i{i}".format(i=i) for i in range(10)]
        expected[5] = "f5_i0"
        self.assertEqual(expected, list(observed.fam.index.values))

    def test_init_duplicated_fid_iid(self):
        """Tests the creation of the instance, but with duplicated FID/IID."""
        # Modifying the FAM file
        with open(self.prefix + ".fam", "w") as fam:
            print("f0", "i0", 0, 0, 1, -9, file=fam)
            print("f1", "i1", 0, 0, 1, -9, file=fam)
            print("f2", "i2", 0, 0, 2, -9, file=fam)
            print("f3", "i3", 0, 0, 2, -9, file=fam)
            print("f4", "i4", 0, 0, 1, -9, file=fam)
            print("f0", "i0", 0, 0, 2, -9, file=fam)
            print("f6", "i6", 0, 0, 1, -9, file=fam)
            print("f7", "i7", 0, 0, 1, -9, file=fam)
            print("f8", "i8", 0, 0, 1, -9, file=fam)
            print("f9", "i9", 0, 0, 2, -9, file=fam)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            PlinkGenotypes(self.prefix)

    def test_repr(self):
        """Tests the '__repr__' function (and as context manager)."""
        with PlinkGenotypes(self.prefix) as plink_geno:
            self.assertEqual(
                "PlinkGenotypes(10 samples; 4 markers)",
                str(plink_geno),
            )

    def test_get_genotypes_additive(self):
        """Tests the 'get_genotypes' function (additive)."""
        observed = PlinkGenotypes(self.prefix)

        # Checking marker_1
        marker_1 = observed.get_genotypes("marker_1")
        self.assertTrue(isinstance(marker_1, MarkerGenotypes))
        self.assertEqual("marker_1", marker_1.marker)
        self.assertEqual(1, marker_1.chrom)
        self.assertEqual(1, marker_1.pos)
        self.assertEqual("C", marker_1.minor)
        self.assertEqual("T", marker_1.major)
        self.assertTrue(self.expected_marker_1_add.equals(marker_1.genotypes))

        # Checking marker_2
        marker_2 = observed.get_genotypes("marker_2")
        self.assertTrue(isinstance(marker_2, MarkerGenotypes))
        self.assertEqual("marker_2", marker_2.marker)
        self.assertEqual(1, marker_2.chrom)
        self.assertEqual(2, marker_2.pos)
        self.assertEqual("G", marker_2.minor)
        self.assertEqual("C", marker_2.major)
        self.assertTrue(self.expected_marker_2_add.equals(marker_2.genotypes))

        # Checking marker_3
        marker_3 = observed.get_genotypes("marker_3")
        self.assertTrue(isinstance(marker_3, MarkerGenotypes))
        self.assertEqual("marker_3", marker_3.marker)
        self.assertEqual(2, marker_3.chrom)
        self.assertEqual(100, marker_3.pos)
        self.assertEqual("T", marker_3.minor)
        self.assertEqual("A", marker_3.major)
        self.assertTrue(self.expected_marker_3_add.equals(marker_3.genotypes))

        # Checking marker_4
        marker_4 = observed.get_genotypes("marker_4")
        self.assertTrue(isinstance(marker_4, MarkerGenotypes))
        self.assertEqual("marker_4", marker_4.marker)
        self.assertEqual(3, marker_4.chrom)
        self.assertEqual(230, marker_4.pos)
        self.assertEqual("G", marker_4.minor)
        self.assertEqual("T", marker_4.major)
        self.assertTrue(self.expected_marker_4_add.equals(marker_4.genotypes))

    def test_get_genotypes_genotypic(self):
        """Tests the 'get_genotypes' function (genotypic)."""
        observed = PlinkGenotypes(self.prefix)

        # Checking marker_1
        marker_1 = observed.get_genotypes("marker_1", Representation.GENOTYPIC)
        self.assertTrue(isinstance(marker_1, MarkerGenotypes))
        self.assertEqual("marker_1", marker_1.marker)
        self.assertEqual(1, marker_1.chrom)
        self.assertEqual(1, marker_1.pos)
        self.assertEqual("C", marker_1.minor)
        self.assertEqual("T", marker_1.major)
        self.assertTrue(self.expected_marker_1_geno.equals(marker_1.genotypes))

        # Checking marker_2
        marker_2 = observed.get_genotypes("marker_2", Representation.GENOTYPIC)
        self.assertTrue(isinstance(marker_2, MarkerGenotypes))
        self.assertEqual("marker_2", marker_2.marker)
        self.assertEqual(1, marker_2.chrom)
        self.assertEqual(2, marker_2.pos)
        self.assertEqual("G", marker_2.minor)
        self.assertEqual("C", marker_2.major)
        self.assertTrue(self.expected_marker_2_geno.equals(marker_2.genotypes))

        # Checking marker_3
        marker_3 = observed.get_genotypes("marker_3", Representation.GENOTYPIC)
        self.assertTrue(isinstance(marker_3, MarkerGenotypes))
        self.assertEqual("marker_3", marker_3.marker)
        self.assertEqual(2, marker_3.chrom)
        self.assertEqual(100, marker_3.pos)
        self.assertEqual("T", marker_3.minor)
        self.assertEqual("A", marker_3.major)
        self.assertTrue(self.expected_marker_3_geno.equals(marker_3.genotypes))

        # Checking marker_4
        marker_4 = observed.get_genotypes("marker_4", Representation.GENOTYPIC)
        self.assertTrue(isinstance(marker_4, MarkerGenotypes))
        self.assertEqual("marker_4", marker_4.marker)
        self.assertEqual(3, marker_4.chrom)
        self.assertEqual(230, marker_4.pos)
        self.assertEqual("G", marker_4.minor)
        self.assertEqual("T", marker_4.major)
        self.assertTrue(self.expected_marker_4_geno.equals(marker_4.genotypes))

    def test_get_genotypes_dosage(self):
        """Tests the 'get_genotypes' function (dosage)."""
        observed = PlinkGenotypes(self.prefix)
        with self.assertRaises(ValueError) as cm:
            observed.get_genotypes("marker_1", Representation.DOSAGE)
        self.assertEqual(
            "DOSAGE is an invalid representation for genotyped data (it is "
            "usually used for imputed data)",
            str(cm.exception),
        )

    def test_iter_marker_genotypes_additive(self):
        """Tests the 'iter_marker_genotypes' function (additive)."""
        plink_geno = PlinkGenotypes(self.prefix)

        expected_results = (
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=self.expected_marker_1_add),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=self.expected_marker_2_add),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=self.expected_marker_3_add),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=self.expected_marker_4_add),
        )
        zipped = zip(plink_geno.iter_marker_genotypes(), expected_results)
        for observed, expected in zipped:
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_genotypic(self):
        """Tests the 'iter_marker_genotypes' function (genotypic)."""
        plink_geno = PlinkGenotypes(self.prefix)

        expected_results = (
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=self.expected_marker_1_geno),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=self.expected_marker_2_geno),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=self.expected_marker_3_geno),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=self.expected_marker_4_geno),
        )
        zipped = zip(
            plink_geno.iter_marker_genotypes(Representation.GENOTYPIC),
            expected_results,
        )
        for observed, expected in zipped:
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.chrom, observed.chrom)
            self.assertEqual(expected.pos, observed.pos)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_dosage(self):
        """Tests the 'get_genotypes' function (dosage)."""
        observed = PlinkGenotypes(self.prefix)
        with self.assertRaises(ValueError) as cm:
            next(observed.iter_marker_genotypes(Representation.DOSAGE))
        self.assertEqual(
            "DOSAGE is an invalid representation for genotyped data (it is "
            "usually used for imputed data)",
            str(cm.exception),
        )
