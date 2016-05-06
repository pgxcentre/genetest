

# This file is part of project_x.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.


import os
import random
import unittest
from tempfile import TemporaryDirectory
from itertools import zip_longest as zip

import numpy as np
import pandas as pd
from pyplink import PyPlink
from pkg_resources import resource_filename

from ..genotypes.core import GenotypesContainer, Representation, \
                             MarkerGenotypes
from ..genotypes.plink import PlinkGenotypes
from ..genotypes.impute2 import Impute2Genotypes, get_index
from ..genotypes.vcf import VCFGenotypes


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

    def test_encode_chrom(self):
        """Tests the 'encode_chrom' function."""
        chromosomes = ["chr1", 1, 2, "3", "X", "x", "y", "XY", "M", "Mt", "yx"]
        expected_chrom = [1, 1, 2, 3, 23, 23, 24, 25, 26, 26, 25]
        for expected, chrom in zip(expected_chrom, chromosomes):
            self.assertEqual(expected, PlinkGenotypes.encode_chrom(chrom))

    def test_encode_invalid_chrom(self):
        """Tests the 'encode_chrom' function with an invalid chromosome."""
        with self.assertRaises(ValueError) as cm:
            PlinkGenotypes.encode_chrom("invalid_chr")
        self.assertEqual("INVALID_CHR: invalid chromosome", str(cm.exception))


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
            print("f0 i0 0 0 1 -9", file=fam)
            print("f1 i1 0 0 1 -9", file=fam)
            print("f2 i2 0 0 2 -9", file=fam)
            print("f3 i3 0 0 2 -9", file=fam)
            print("f4 i4 0 0 1 -9", file=fam)
            print("f5 i5 0 0 2 -9", file=fam)
            print("f6 i6 0 0 1 -9", file=fam)
            print("f7 i7 0 0 1 -9", file=fam)
            print("f8 i8 0 0 1 -9", file=fam)
            print("f9 i9 0 0 2 -9", file=fam)

        # The expected results (additive)
        marker_1_add = pd.DataFrame(
                [("i0", 0), ("i1", 0), ("i2", 0), ("i3", 1), ("i4", 0),
                 ("i5", 0), ("i6", np.nan), ("i7", 2), ("i8", 1), ("i9", 0)],
                columns=["iid", "geno"],
            ).set_index("iid")
        marker_2_add = pd.DataFrame(
                [("i0", 0.0), ("i1", 0.0), ("i2", 0.0), ("i3", 0.0),
                 ("i4", 1.0), ("i5", 1.0), ("i6", 0.0), ("i7", 0.0),
                 ("i8", 0.0), ("i9", 1.0)],
                columns=["iid", "geno"],
            ).set_index("iid")
        marker_3_add = pd.DataFrame(
                [("i0", 0), ("i1", 2), ("i2", 0), ("i3", 0), ("i4", 1),
                 ("i5", 1), ("i6", np.nan), ("i7", 0), ("i8", 0), ("i9", 1)],
                columns=["iid", "geno"],
            ).set_index("iid")
        marker_4_add = pd.DataFrame(
                [("i0", np.nan), ("i1", np.nan), ("i2", np.nan),
                 ("i3", np.nan), ("i4", np.nan), ("i5", np.nan), ("i6",
                 np.nan), ("i7", np.nan), ("i8", np.nan), ("i9", np.nan)],
                columns=["iid", "geno"],
            ).set_index("iid")

        # The expected results (genotypic)
        marker_1_geno = pd.DataFrame(
                [("i0", 0, 0), ("i1", 0, 0), ("i2", 0, 0), ("i3", 1, 0),
                 ("i4", 0, 0), ("i5", 0, 0), ("i6", np.nan, np.nan),
                 ("i7", 0, 1), ("i8", 1, 0), ("i9", 0, 0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        marker_2_geno = pd.DataFrame(
                [("i0", 0.0, 0.0), ("i1", 0.0, 0.0), ("i2", 0.0, 0.0),
                 ("i3", 0.0, 0.0), ("i4", 1.0, 0.0), ("i5", 1.0, 0.0),
                 ("i6", 0.0, 0.0), ("i7", 0.0, 0.0), ("i8", 0.0, 0.0),
                 ("i9", 1.0, 0.0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        marker_3_geno = pd.DataFrame(
                [("i0", 0, 0), ("i1", 0, 1), ("i2", 0, 0), ("i3", 0, 0),
                 ("i4", 1, 0), ("i5", 1, 0), ("i6", np.nan, np.nan),
                 ("i7", 0, 0), ("i8", 0, 0), ("i9", 1, 0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        marker_4_geno = pd.DataFrame(
                [("i0", np.nan, np.nan), ("i1", np.nan, np.nan),
                 ("i2", np.nan, np.nan), ("i3", np.nan, np.nan),
                 ("i4", np.nan, np.nan), ("i5", np.nan, np.nan),
                 ("i6", np.nan, np.nan), ("i7", np.nan, np.nan),
                 ("i8", np.nan, np.nan), ("i9", np.nan, np.nan)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")

        # The expected ADDITIVE results
        self.expected_additive_results = [
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=marker_1_add),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=marker_2_add),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=marker_3_add),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=marker_4_add),
        ]

        # The expected GENOTYPIC results
        self.expected_genotypic_results = [
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=marker_1_geno),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=marker_2_geno),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=marker_3_geno),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=marker_4_geno),
        ]

        # The parameters
        self.parameters = dict(
            prefix=self.prefix,
            representation=Representation.ADDITIVE,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init(self):
        """Tests the creation of a PlinkGenotypes instance."""
        observed = PlinkGenotypes(**self.parameters)

        # We don't check the BED and the BIM file, since they come from
        # PyPlink. We only check the FAM file, because it might be modified.
        self.assertEqual(
            ["i{}".format(i) for i in range(10)],
            list(observed.fam.index.values),
        )

    def test_dosage_representation(self):
        """Tests when the instance is initialize DOSAGE as representation."""
        self.parameters["representation"] = Representation.DOSAGE
        with self.assertRaises(ValueError) as cm:
            PlinkGenotypes(**self.parameters)

        self.assertEqual(
            "DOSAGE is an invalid representation for genotyped data (it is "
            "usually used for imputed data)",
            str(cm.exception),
        )

    def test_init_duplicated_iid(self):
        """Tests the creation of the instance, but with duplicated IID."""
        # Modifying the FAM file
        with open(self.prefix + ".fam", "w") as fam:
            print("f0 i0 0 0 1 -9", file=fam)
            print("f1 i1 0 0 1 -9", file=fam)
            print("f2 i2 0 0 2 -9", file=fam)
            print("f3 i3 0 0 2 -9", file=fam)
            print("f4 i4 0 0 1 -9", file=fam)
            print("f5 i0 0 0 2 -9", file=fam)
            print("f6 i6 0 0 1 -9", file=fam)
            print("f7 i7 0 0 1 -9", file=fam)
            print("f8 i8 0 0 1 -9", file=fam)
            print("f9 i9 0 0 2 -9", file=fam)

        observed = PlinkGenotypes(**self.parameters)

        # The FAM index should now be fid_iid
        expected = ["f{i}_i{i}".format(i=i) for i in range(10)]
        expected[5] = "f5_i0"
        self.assertEqual(expected, list(observed.fam.index.values))

    def test_init_duplicated_fid_iid(self):
        """Tests the creation of the instance, but with duplicated FID/IID."""
        # Modifying the FAM file
        with open(self.prefix + ".fam", "w") as fam:
            print("f0 i0 0 0 1 -9", file=fam)
            print("f1 i1 0 0 1 -9", file=fam)
            print("f2 i2 0 0 2 -9", file=fam)
            print("f3 i3 0 0 2 -9", file=fam)
            print("f4 i4 0 0 1 -9", file=fam)
            print("f0 i0 0 0 2 -9", file=fam)
            print("f6 i6 0 0 1 -9", file=fam)
            print("f7 i7 0 0 1 -9", file=fam)
            print("f8 i8 0 0 1 -9", file=fam)
            print("f9 i9 0 0 2 -9", file=fam)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            PlinkGenotypes(**self.parameters)

    def test_repr(self):
        """Tests the '__repr__' function (and as context manager)."""
        with PlinkGenotypes(**self.parameters) as plink_geno:
            self.assertEqual(
                "PlinkGenotypes(10 samples; 4 markers)",
                str(plink_geno),
            )

    def test_get_genotypes_additive(self):
        """Tests the 'get_genotypes' function (additive)."""
        plink_geno = PlinkGenotypes(**self.parameters)

        random.shuffle(self.expected_additive_results)
        for expected in self.expected_additive_results:
            # Getting the observed results
            observed = plink_geno.get_genotypes(
                marker=expected.marker,
            )

            # Comparing with the expected results
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.chrom, observed.chrom)
            self.assertEqual(expected.pos, observed.pos)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_get_genotypes_genotypic(self):
        """Tests the 'get_genotypes' function (genotypic)."""
        self.parameters["representation"] = Representation.GENOTYPIC
        plink_geno = PlinkGenotypes(**self.parameters)

        random.shuffle(self.expected_genotypic_results)
        for expected in self.expected_genotypic_results:
            # Getting the observed results
            observed = plink_geno.get_genotypes(
                marker=expected.marker,
            )

            # Comparing the expected results
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.chrom, observed.chrom)
            self.assertEqual(expected.pos, observed.pos)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_additive(self):
        """Tests the 'iter_marker_genotypes' function (additive)."""
        plink_geno = PlinkGenotypes(**self.parameters)

        zipped = zip(
            plink_geno.iter_marker_genotypes(),
            self.expected_additive_results,
        )
        for observed, expected in zipped:
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.chrom, observed.chrom)
            self.assertEqual(expected.pos, observed.pos)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_genotypic(self):
        """Tests the 'iter_marker_genotypes' function (genotypic)."""
        self.parameters["representation"] = Representation.GENOTYPIC
        plink_geno = PlinkGenotypes(**self.parameters)

        zipped = zip(
            plink_geno.iter_marker_genotypes(),
            self.expected_genotypic_results,
        )
        for observed, expected in zipped:
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.chrom, observed.chrom)
            self.assertEqual(expected.pos, observed.pos)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))


class TestImpute2(unittest.TestCase):
    def setUp(self):
        self.tmpdir = TemporaryDirectory(prefix="project_x_")

        # Creating a sample file
        sample_file = os.path.join(self.tmpdir.name, "input.sample")
        with open(sample_file, "w") as f:
            print("ID_1 ID_2 missing father mother sex plink_pheno", file=f)
            print("0 0 0 D D D B", file=f)
            print("f0 i0 0 0 0 1 -9", file=f)
            print("f1 i1 0 0 0 1 -9", file=f)
            print("f2 i2 0 0 0 2 -9", file=f)
            print("f3 i3 0 0 0 2 -9", file=f)
            print("f4 i4 0 0 0 1 -9", file=f)
            print("f5 i5 0 0 0 2 -9", file=f)
            print("f6 i6 0 0 0 1 -9", file=f)
            print("f7 i7 0 0 0 1 -9", file=f)
            print("f8 i8 0 0 0 1 -9", file=f)
            print("f9 i9 0 0 0 2 -9", file=f)

        # Creating an IMPUTE2 file
        impute2_file = os.path.join(self.tmpdir.name, "input.impute2")
        with open(impute2_file, "w") as f:
            print("1 marker_1 1 T C 0.9 0.1 0 0.99 0.01 0 1 0 0 0 0.91 0.09 "
                  "1 0 0 1 0 0 0.8 0.2 0 0 0.1 0.9 0.01 0.98 0.01 "
                  "1 0 0", file=f)
            print("1 marker_2 2 C G 0.98 0.02 0 0.99 0.01 0 1 0 0 1 0 0 "
                  "0 0.9 0.1 0.1 0.9 0 1 0 0 0.94 0.06 0 1 0 0 "
                  "0.03 0.97 0", file=f)
            print("2 marker_3 100 T A 0 0.1 0.9 0.93 0.07 0 0 0.04 0.96 0 0 1 "
                  "0.05 0.95 0 0 0.95 0.05 0 0.3 0.7 0 0 1 0 0 1 "
                  "0 1 0", file=f)
            print("3 marker_4 230 T G 0 0.3 0.7 0.1 0.8 0.1 0.7 0.2 0.1 "
                  "0.15 0.7 0.15 0.8 0.1 0 0 0.2 0.8 0 0.8 0.2 0.2 0.8 0 "
                  "0.8 0.2 0 0.5 0.4 0.1", file=f)

        # Generating the index
        get_index(impute2_file, cols=[0, 1, 2], sep=" ",
                  names=["chrom", "name", "pos"])

        # The expected results (dosage)
        marker_1_dose = pd.DataFrame(
            [("i0", 0.1), ("i1", 0.01), ("i2", 0), ("i3", 1.09), ("i4", 0),
             ("i5", 0), ("i6", np.nan), ("i7", 1.9), ("i8", 1.0), ("i9", 0)],
            columns=["iid", "geno"],
        ).set_index("iid")
        marker_2_dose = pd.DataFrame(
            [("i0", 0.02), ("i1", 0.01), ("i2", 0), ("i3", 0), ("i4", 1.1),
             ("i5", 0.9), ("i6", 0), ("i7", 0.06), ("i8", 0), ("i9", 0.97)],
            columns=["iid", "geno"],
            ).set_index("iid")
        marker_3_dose = pd.DataFrame(
            [("i0", 0.1), ("i1", 1.93), ("i2", 0.04), ("i3", 0), ("i4", 1.05),
             ("i5", 0.95), ("i6", np.nan), ("i7", 0), ("i8", 0), ("i9", 1)],
            columns=["iid", "geno"],
        ).set_index("iid")
        marker_4_dose = pd.DataFrame(
            [("i0", np.nan), ("i1", np.nan), ("i2", np.nan), ("i3", np.nan),
             ("i4", np.nan), ("i5", np.nan), ("i6", np.nan), ("i7", np.nan),
             ("i8", np.nan), ("i9", np.nan)],
            columns=["iid", "geno"],
        ).set_index("iid")

        # The expected results (additive)
        marker_1_add = pd.DataFrame(
                [("i0", 0), ("i1", 0), ("i2", 0), ("i3", 1), ("i4", 0),
                 ("i5", 0), ("i6", np.nan), ("i7", 2), ("i8", 1), ("i9", 0)],
                columns=["iid", "geno"],
            ).set_index("iid")
        marker_2_add = pd.DataFrame(
                [("i0", 0.0), ("i1", 0.0), ("i2", 0.0), ("i3", 0.0),
                 ("i4", 1.0), ("i5", 1.0), ("i6", 0.0), ("i7", 0.0),
                 ("i8", 0.0), ("i9", 1.0)],
                columns=["iid", "geno"],
            ).set_index("iid")
        marker_3_add = pd.DataFrame(
                [("i0", 0), ("i1", 2), ("i2", 0), ("i3", 0), ("i4", 1),
                 ("i5", 1), ("i6", np.nan), ("i7", 0), ("i8", 0), ("i9", 1)],
                columns=["iid", "geno"],
            ).set_index("iid")
        marker_4_add = pd.DataFrame(
                [("i0", np.nan), ("i1", np.nan), ("i2", np.nan),
                 ("i3", np.nan), ("i4", np.nan), ("i5", np.nan), ("i6",
                 np.nan), ("i7", np.nan), ("i8", np.nan), ("i9", np.nan)],
                columns=["iid", "geno"],
            ).set_index("iid")

        # The expected results (genotypic)
        marker_1_geno = pd.DataFrame(
                [("i0", 0, 0), ("i1", 0, 0), ("i2", 0, 0), ("i3", 1, 0),
                 ("i4", 0, 0), ("i5", 0, 0), ("i6", np.nan, np.nan),
                 ("i7", 0, 1), ("i8", 1, 0), ("i9", 0, 0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        marker_2_geno = pd.DataFrame(
                [("i0", 0.0, 0.0), ("i1", 0.0, 0.0), ("i2", 0.0, 0.0),
                 ("i3", 0.0, 0.0), ("i4", 1.0, 0.0), ("i5", 1.0, 0.0),
                 ("i6", 0.0, 0.0), ("i7", 0.0, 0.0), ("i8", 0.0, 0.0),
                 ("i9", 1.0, 0.0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        marker_3_geno = pd.DataFrame(
                [("i0", 0, 0), ("i1", 0, 1), ("i2", 0, 0), ("i3", 0, 0),
                 ("i4", 1, 0), ("i5", 1, 0), ("i6", np.nan, np.nan),
                 ("i7", 0, 0), ("i8", 0, 0), ("i9", 1, 0)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")
        marker_4_geno = pd.DataFrame(
                [("i0", np.nan, np.nan), ("i1", np.nan, np.nan),
                 ("i2", np.nan, np.nan), ("i3", np.nan, np.nan),
                 ("i4", np.nan, np.nan), ("i5", np.nan, np.nan),
                 ("i6", np.nan, np.nan), ("i7", np.nan, np.nan),
                 ("i8", np.nan, np.nan), ("i9", np.nan, np.nan)],
                columns=["iid", "geno_ab", "geno_bb"],
            ).set_index("iid")

        # The expected DOSAGE ressults
        self.expected_dosage_results = [
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=marker_1_dose),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=marker_2_dose),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=marker_3_dose),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=marker_4_dose),
        ]

        # The expected ADDITIVE results
        self.expected_additive_results = [
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=marker_1_add),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=marker_2_add),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=marker_3_add),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=marker_4_add),
        ]

        # The expected GENOTYPIC results
        self.expected_genotypic_results = [
            MarkerGenotypes(marker="marker_1", minor="C", major="T", chrom=1,
                            pos=1, genotypes=marker_1_geno),
            MarkerGenotypes(marker="marker_2", minor="G", major="C", chrom=1,
                            pos=2, genotypes=marker_2_geno),
            MarkerGenotypes(marker="marker_3", minor="T", major="A", chrom=2,
                            pos=100, genotypes=marker_3_geno),
            MarkerGenotypes(marker="marker_4", minor="G", major="T", chrom=3,
                            pos=230, genotypes=marker_4_geno),
        ]

        # The Impute2Genotypes parameters
        self.parameters = dict(
            filename=impute2_file,
            sample_filename=sample_file,
            representation=Representation.DOSAGE,
            probability_threshold=0.9,
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init(self):
        """Tests the creation of a Impute2Genotypes instance (with index)."""
        with Impute2Genotypes(**self.parameters) as observed:
            # We only check the samples, because they might be modified.
            self.assertEqual(
                ["i{}".format(i) for i in range(10)],
                list(observed.samples.index.values),
            )

    def test_init_no_index(self):
        """Tests the creation of a Impute2Genotypes instance (no index)."""
        os.remove(self.parameters["filename"] + ".idx")
        with Impute2Genotypes(**self.parameters) as observed:
            # We only check the samples, because they might be modified.
            self.assertEqual(
                ["i{}".format(i) for i in range(10)],
                list(observed.samples.index.values),
            )

    def test_init_duplicated_iid(self):
        """Tests the creation of the instance, but with duplicated IID."""
        with open(self.parameters["sample_filename"], "w") as f:
            print("ID_1 ID_2 missing father mother sex plink_pheno", file=f)
            print("0 0 0 D D D B", file=f)
            print("f0 i0 0 0 0 1 -9", file=f)
            print("f1 i1 0 0 0 1 -9", file=f)
            print("f2 i2 0 0 0 2 -9", file=f)
            print("f3 i3 0 0 0 2 -9", file=f)
            print("f4 i4 0 0 0 1 -9", file=f)
            print("f5 i0 0 0 0 2 -9", file=f)
            print("f6 i6 0 0 0 1 -9", file=f)
            print("f7 i7 0 0 0 1 -9", file=f)
            print("f8 i8 0 0 0 1 -9", file=f)
            print("f9 i9 0 0 0 2 -9", file=f)

        expected = ["f{i}_i{i}".format(i=i) for i in range(10)]
        expected[5] = "f5_i0"

        with Impute2Genotypes(**self.parameters) as observed:
            # We only check the samples, because they might be modified.
            self.assertEqual(expected, list(observed.samples.index.values))

    def test_init_duplicated_fid_iid(self):
        """Tests the creation of the instance, but with duplicated FID/IID."""
        with open(self.parameters["sample_filename"], "w") as f:
            print("ID_1 ID_2 missing father mother sex plink_pheno", file=f)
            print("0 0 0 D D D B", file=f)
            print("f0 i0 0 0 0 1 -9", file=f)
            print("f1 i1 0 0 0 1 -9", file=f)
            print("f2 i2 0 0 0 2 -9", file=f)
            print("f3 i3 0 0 0 2 -9", file=f)
            print("f4 i4 0 0 0 1 -9", file=f)
            print("f0 i0 0 0 0 2 -9", file=f)
            print("f6 i6 0 0 0 1 -9", file=f)
            print("f7 i7 0 0 0 1 -9", file=f)
            print("f8 i8 0 0 0 1 -9", file=f)
            print("f9 i9 0 0 0 2 -9", file=f)

        # This should raise a ValueError
        with self.assertRaises(ValueError):
            with Impute2Genotypes(**self.parameters):
                pass

    def test_repr(self):
        """Tests the '__repr__' function."""
        with Impute2Genotypes(**self.parameters) as geno:
            self.assertEqual(
                "Impute2Genotypes(10 samples, 4 markers)",
                str(geno),
            )

    def test_get_genotypes_dosage(self):
        """Tests the 'get_genotypes' function (dosage)."""
        random.shuffle(self.expected_dosage_results)
        with Impute2Genotypes(**self.parameters) as imp_geno:
            for expected in self.expected_dosage_results:
                # Getting the observed results
                observed = imp_geno.get_genotypes(
                    marker=expected.marker,
                )

                # Comparing with the expected results
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertEqual(expected.genotypes.index.tolist(),
                                 observed.genotypes.index.tolist())
                self.assertEqual(expected.genotypes.columns.tolist(),
                                 observed.genotypes.columns.tolist())
                np.testing.assert_allclose(expected.genotypes,
                                           observed.genotypes)

    def test_get_genotypes_additive(self):
        """Tests the 'get_genotypes' function (additive)."""
        self.parameters["representation"] = Representation.ADDITIVE
        random.shuffle(self.expected_additive_results)
        with Impute2Genotypes(**self.parameters) as imp_geno:
            for expected in self.expected_additive_results:
                # Getting the observed results
                observed = imp_geno.get_genotypes(
                    marker=expected.marker,
                )

                # Comparing with the expected results
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_get_genotypes_genotypic(self):
        """Tests the 'get_genotypes' function (genotypic)."""
        self.parameters["representation"] = Representation.GENOTYPIC
        random.shuffle(self.expected_genotypic_results)
        with Impute2Genotypes(**self.parameters) as imp_geno:
            for expected in self.expected_genotypic_results:
                # Getting the observed results
                observed = imp_geno.get_genotypes(
                    marker=expected.marker,
                )

                # Comparing with the expected results
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_dosage(self):
        """Tests the 'iter_marker_genotypes' function (dosage)."""
        with Impute2Genotypes(**self.parameters) as imp_geno:
            zipped = zip(
                imp_geno.iter_marker_genotypes(),
                self.expected_dosage_results,
            )
            for observed, expected in zipped:
                # Comparing with the expected results
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertEqual(expected.genotypes.index.tolist(),
                                 observed.genotypes.index.tolist())
                self.assertEqual(expected.genotypes.columns.tolist(),
                                 observed.genotypes.columns.tolist())
                np.testing.assert_allclose(expected.genotypes,
                                           observed.genotypes)

    def test_iter_marker_genotypes_additive(self):
        """Tests the 'iter_marker_genotypes' function (additive)."""
        self.parameters["representation"] = Representation.ADDITIVE
        with Impute2Genotypes(**self.parameters) as imp_geno:
            zipped = zip(
                imp_geno.iter_marker_genotypes(),
                self.expected_additive_results,
            )
            for observed, expected in zipped:
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_genotypic(self):
        """Tests the 'iter_marker_genotypes' function (genotypic)."""
        self.parameters["representation"] = Representation.GENOTYPIC
        with Impute2Genotypes(**self.parameters) as imp_geno:
            zipped = zip(
                imp_geno.iter_marker_genotypes(),
                self.expected_genotypic_results,
            )
            for observed, expected in zipped:
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_different_probability_threshold(self):
        """Tests with a different probability threshold."""
        self.parameters["probability_threshold"] = 0.95
        with Impute2Genotypes(**self.parameters) as imp_geno:
            # Modifying the expected results
            expected = self.expected_dosage_results[0]
            expected.genotypes.iloc[[0, 3, 7], :] = np.nan

            # Getting the results
            observed = imp_geno.get_genotypes(
                marker=expected.marker,
            )

            # Comparing the results
            self.assertTrue(isinstance(observed, MarkerGenotypes))
            self.assertEqual(expected.marker, observed.marker)
            self.assertEqual(expected.chrom, observed.chrom)
            self.assertEqual(expected.pos, observed.pos)
            self.assertEqual(expected.minor, observed.minor)
            self.assertEqual(expected.major, observed.major)
            self.assertTrue(expected.genotypes.equals(observed.genotypes))


class TestVCF(unittest.TestCase):
    def setUp(self):
        vcf_file = resource_filename(__name__, "data/genotypes/input.vcf.gz")

        # The expected results (additive)
        marker_1_add = pd.DataFrame(
                [("s1", 2.0), ("s2", 1.0), ("s3", 0.0), ("s4", 0.0)],
                columns=["SampleID", "geno"],
            ).set_index("SampleID")
        marker_2_add = pd.DataFrame(
                [("s1", 0.0), ("s2", np.nan), ("s3", 0.0), ("s4", 1.0)],
                columns=["SampleID", "geno"],
            ).set_index("SampleID")
        marker_3_add = pd.DataFrame(
                [("s1", 0.0), ("s2", 1.0), ("s3", 1.0), ("s4", np.nan)],
                columns=["SampleID", "geno"],
            ).set_index("SampleID")
        marker_4_add = pd.DataFrame(
                [("s1", 1.0), ("s2", 2.0), ("s3", 0.0), ("s4", 0.0)],
                columns=["SampleID", "geno"],
            ).set_index("SampleID")

        # The expected results (genotypic)
        marker_1_geno = pd.DataFrame(
                [("s1", 0.0, 1.0), ("s2", 1.0, 0.0), ("s3", 0.0, 0.0),
                 ("s4", 0.0, 0.0)],
                columns=["SampleID", "geno_ab", "geno_bb"],
            ).set_index("SampleID")
        marker_2_geno = pd.DataFrame(
                [("s1", 0.0, 0.0), ("s2", np.nan, np.nan), ("s3", 0.0, 0.0),
                 ("s4", 1.0, 0.0)],
                columns=["SampleID", "geno_ab", "geno_bb"],
            ).set_index("SampleID")
        marker_3_geno = pd.DataFrame(
                [("s1", 0.0, 0.0), ("s2", 1.0, 0.0), ("s3", 1.0, 0.0),
                 ("s4", np.nan, np.nan)],
                columns=["SampleID", "geno_ab", "geno_bb"],
            ).set_index("SampleID")
        marker_4_geno = pd.DataFrame(
                [("s1", 1.0, 0.0), ("s2", 0.0, 1.0), ("s3", 0.0, 0.0),
                 ("s4", 0.0, 0.0)],
                columns=["SampleID", "geno_ab", "geno_bb"],
            ).set_index("SampleID")

        # The expected ADDITIVE results
        self.expected_additive_results = [
            MarkerGenotypes(marker="1:1", minor="G", major="A", chrom=1,
                            pos=1, genotypes=marker_1_add),
            MarkerGenotypes(marker="marker_2", minor="C", major="T", chrom=1,
                            pos=2, genotypes=marker_2_add),
            MarkerGenotypes(marker="marker_3", minor="A", major="T", chrom=2,
                            pos=100, genotypes=marker_3_add),
            MarkerGenotypes(marker="marker_4", minor="A", major="C", chrom=3,
                            pos=230, genotypes=marker_4_add),
        ]

        # The expected GENOTYPIC results
        self.expected_genotypic_results = [
            MarkerGenotypes(marker="1:1", minor="G", major="A", chrom=1,
                            pos=1, genotypes=marker_1_geno),
            MarkerGenotypes(marker="marker_2", minor="C", major="T", chrom=1,
                            pos=2, genotypes=marker_2_geno),
            MarkerGenotypes(marker="marker_3", minor="A", major="T", chrom=2,
                            pos=100, genotypes=marker_3_geno),
            MarkerGenotypes(marker="marker_4", minor="A", major="C", chrom=3,
                            pos=230, genotypes=marker_4_geno),
        ]

        # The Impute2Genotypes parameters
        self.parameters = dict(
            filename=vcf_file,
            representation=Representation.ADDITIVE,
        )

    def test_init(self):
        """Tests the creation of a VCFGenotypes instance."""
        # Creating a new object
        with VCFGenotypes(**self.parameters) as vcf_geno:
            samples = vcf_geno.samples

        expected_samples = pd.Index(["s1", "s2", "s3", "s4"], name="SampleID")
        self.assertTrue(expected_samples.equals(samples))

    def test_repr(self):
        """Tests the '__repr__' function."""
        with VCFGenotypes(**self.parameters) as geno:
            self.assertEqual(
                "VCFGenotypes(4 samples)",
                str(geno),
            )

    def test_dosage_representation(self):
        """Tests when the instance is initialize DOSAGE as representation."""
        self.parameters["representation"] = Representation.DOSAGE
        with self.assertRaises(ValueError) as cm:
            VCFGenotypes(**self.parameters)

        self.assertEqual(
            "DOSAGE is an invalid representation for sequenced data (it is "
            "usually used for imputed data)",
            str(cm.exception),
        )

    def test_get_invalid_genotype(self):
        """Tests when asking for a missing marker."""
        with VCFGenotypes(**self.parameters) as vcf_geno:
            with self.assertRaises(ValueError) as cm:
                vcf_geno.get_genotypes("chr1", 204123)
        self.assertEqual(
            "no marker positioned on chromosome chr1, position 204123",
            str(cm.exception),
        )

    def test_get_multiallele_genotype(self):
        """Tests when asking for a marker with more than two alleles."""
        with VCFGenotypes(**self.parameters) as vcf_geno:
            with self.assertRaises(ValueError) as cm:
                vcf_geno.get_genotypes("chr3", 240)
        self.assertEqual(
            "chr3: 240: more than two alleles",
            str(cm.exception),
        )

    def test_get_genotypes_additive(self):
        """Tests the 'get_genotypes' function (additive)."""
        random.shuffle(self.expected_additive_results)
        with VCFGenotypes(**self.parameters) as vcf_geno:
            for expected in self.expected_additive_results:
                # Getting the observed results
                observed = vcf_geno.get_genotypes(
                    chrom="chr{}".format(expected.chrom),
                    pos=expected.pos,
                )

                # Comparing with the expected results
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_get_genotypes_genotypic(self):
        """Tests the 'get_genotypes' function (genotypic)."""
        self.parameters["representation"] = Representation.GENOTYPIC
        random.shuffle(self.expected_genotypic_results)
        with VCFGenotypes(**self.parameters) as vcf_geno:
            for expected in self.expected_genotypic_results:
                # Getting the observed results
                observed = vcf_geno.get_genotypes(
                    chrom="chr{}".format(expected.chrom),
                    pos=expected.pos,
                )

                # Comparing with the expected results
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_additive(self):
        """Tests the 'iter_marker_genotypes' function (additive)."""
        with VCFGenotypes(**self.parameters) as vcf_geno:
            zipped = zip(
                vcf_geno.iter_marker_genotypes(),
                self.expected_additive_results,
            )
            for observed, expected in zipped:
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_iter_marker_genotypes_genotypic(self):
        """Tests the 'iter_marker_genotypes' function (genotypic)."""
        self.parameters["representation"] = Representation.GENOTYPIC
        with VCFGenotypes(**self.parameters) as vcf_geno:
            zipped = zip(
                vcf_geno.iter_marker_genotypes(),
                self.expected_genotypic_results,
            )
            for observed, expected in zipped:
                self.assertTrue(isinstance(observed, MarkerGenotypes))
                self.assertEqual(expected.marker, observed.marker)
                self.assertEqual(expected.chrom, observed.chrom)
                self.assertEqual(expected.pos, observed.pos)
                self.assertEqual(expected.minor, observed.minor)
                self.assertEqual(expected.major, observed.major)
                self.assertTrue(expected.genotypes.equals(observed.genotypes))

    def test_fetch_before_iter(self):
        """Tests when fetching before iterating."""
        with VCFGenotypes(**self.parameters) as vcf_geno:
            var = vcf_geno.get_genotypes("chr3", 230)
            self.assertEqual("marker_4", var.marker)
            var = next(vcf_geno.iter_marker_genotypes())
            self.assertEqual("1:1", var.marker)
            var = vcf_geno.get_genotypes("chr2", 100)
            self.assertEqual("marker_3", var.marker)
            var = next(vcf_geno.iter_marker_genotypes())
            self.assertEqual("1:1", var.marker)
