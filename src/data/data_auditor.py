from collections import Counter
from pathlib import Path
from statistics import mean, median
import numpy as np
import statistics

from PIL import Image


class DatasetAuditor:

    SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
    SUPPORTED_MASK_EXTENSIONS = {".png"}

    def __init__(self, dataset_root: str | Path):

        self.dataset_root = Path(dataset_root)

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_root}"
            )

        # ---------------------------------------------------------
        # Dataset
        # ---------------------------------------------------------

        self.total_classes = 0
        self.total_samples = 0

        self.class_distribution = Counter()

        # ---------------------------------------------------------
        # Pair Integrity
        # ---------------------------------------------------------

        self.total_images = 0
        self.total_masks = 0

        self.matched_pairs = 0

        self.missing_images = []
        self.missing_masks = []

        self.orphan_images = []
        self.orphan_masks = []

        self.duplicate_sample_ids = []

        # ---------------------------------------------------------
        # Original Images
        # ---------------------------------------------------------

        self.image_sizes = Counter()
        self.image_modes = Counter()
        self.image_formats = Counter()

        self.widths = []
        self.heights = []
        self.aspect_ratios = []

        # ---------------------------------------------------------
        # Masks
        # ---------------------------------------------------------

        self.mask_modes = Counter()
        self.mask_formats = Counter()

        # Binary statistics
        self.binary_masks = 0
        self.non_binary_masks = 0

        # Dataset characteristics
        self.no_defect_masks = 0
        self.full_masks = 0

        # Geometry
        self.mask_coverages = []

        # Integrity
        self.resolution_mismatches = []

        # Pixel statistics
        self.unique_mask_value_sets = Counter()

        # ---------------------------------------------------------
        # Audit Status
        # ---------------------------------------------------------

        self.audit_issues = []

        self.dataset_ready = True


        # ---------------------------------------------------------
        # Integrity
        # ---------------------------------------------------------

        self.corrupted_images = []
        self.corrupted_masks = []

    # -------------------------------------------------- #
    # Public API
    # -------------------------------------------------- #

    def audit(self):

        self._scan_dataset()

        self._print_report()

    # -------------------------------------------------- #
    # Dataset Scan
    # -------------------------------------------------- #

    def _scan_dataset(self) -> None:
        """
        Scan the complete dataset.

        Responsibilities
        ----------------
        - Discover all class folders
        - Scan every class independently
        - Verify overall dataset integrity
        """

        for class_dir in sorted(self.dataset_root.iterdir()):

            if not class_dir.is_dir():
                continue

            self._scan_class(class_dir)

        self._verify_pairs()


    def _scan_class(self, class_dir: Path) -> None:
        """
        Scan a single class directory and validate image-mask pairs.

        Expected Structure
        ------------------
        MT_Crack/
            Imgs/
                exp1_num_0001.jpg
                exp1_num_0001.png
                exp1_num_0002.jpg
                exp1_num_0002.png
        """

        imgs_dir = class_dir / "Imgs"

        if not imgs_dir.exists():
            return

        self.total_classes += 1

        jpg_files: dict[str, Path] = {}
        png_files: dict[str, Path] = {}

        # --------------------------------------------------
        # Collect JPG and PNG files
        # --------------------------------------------------

        for file_path in imgs_dir.iterdir():

            if not file_path.is_file():
                continue

            suffix = file_path.suffix.lower()
            sample_id = file_path.stem

            if suffix in self.SUPPORTED_IMAGE_EXTENSIONS:
                jpg_files[sample_id] = file_path

            elif suffix in self.SUPPORTED_MASK_EXTENSIONS:
                png_files[sample_id] = file_path

        # --------------------------------------------------
        # Verify pairs
        # --------------------------------------------------

        sample_ids = sorted(set(jpg_files.keys()) | set(png_files.keys()))

        class_sample_count = 0

        for sample_id in sample_ids:

            image_path = jpg_files.get(sample_id)
            mask_path = png_files.get(sample_id)

            # -------------------------
            # Valid pair
            # -------------------------

            if image_path and mask_path:

                self.matched_pairs += 1
                self.total_samples += 1

                self.total_images += 1
                self.total_masks += 1

                class_sample_count += 1

                # Analyze separately
                self._analyze_original_image(image_path)
                self._analyze_mask(image_path, mask_path)

            # -------------------------
            # Missing image
            # -------------------------

            elif mask_path:

                self.missing_images.append(mask_path)

            # -------------------------
            # Missing mask
            # -------------------------

            elif image_path:

                self.missing_masks.append(image_path)

        self.class_distribution[class_dir.name] = class_sample_count



    def _analyze_original_image(self, image_path: Path) -> None:
        """
        Analyze a single original inspection image.

        Collects
        --------
        - Resolution
        - Width
        - Height
        - Aspect Ratio
        - Image Mode
        - Image Format
        """

        try:

            with Image.open(image_path) as img:

                width, height = img.size

                # ------------------------------------------
                # Resolution Statistics
                # ------------------------------------------

                self.image_sizes[(width, height)] += 1

                self.widths.append(width)

                self.heights.append(height)

                self.aspect_ratios.append(width / height)

                # ------------------------------------------
                # Image Properties
                # ------------------------------------------

                self.image_modes[img.mode] += 1

                self.image_formats[img.format] += 1

        except Exception:

            self.corrupted_images.append(image_path)



    def _analyze_mask(self, image_path: Path, mask_path: Path) -> None:
        """
        Analyze a segmentation mask.

        Collects
        --------
        - Resolution consistency
        - Mask mode & format
        - Binary / non-binary statistics
        - Unique pixel value distribution
        - Defect coverage
        - Background-only masks
        - Completely filled masks
        """

        try:

            with Image.open(image_path) as image, Image.open(mask_path) as mask:

                # --------------------------------------------------
                # Resolution Consistency
                # --------------------------------------------------

                if image.size != mask.size:

                    self.resolution_mismatches.append(
                        {
                            "image": image_path,
                            "mask": mask_path,
                            "image_size": image.size,
                            "mask_size": mask.size,
                        }
                    )

                # --------------------------------------------------
                # Mask Properties
                # --------------------------------------------------

                self.mask_modes[mask.mode] += 1
                self.mask_formats[mask.format] += 1

                # --------------------------------------------------
                # Convert to NumPy
                # --------------------------------------------------

                mask_array = np.asarray(mask)

                unique_values = np.unique(mask_array)

                # Store unique value pattern
                self.unique_mask_value_sets[
                    tuple(unique_values.tolist())
                ] += 1

                # --------------------------------------------------
                # Binary Statistics
                # --------------------------------------------------

                if set(unique_values).issubset({0, 255}):

                    self.binary_masks += 1

                else:

                    self.non_binary_masks += 1

                # --------------------------------------------------
                # Background-only / Full Mask
                # --------------------------------------------------

                if np.all(mask_array == 0):

                    self.no_defect_masks += 1

                elif np.all(mask_array == 255):

                    self.full_masks += 1

                # --------------------------------------------------
                # Defect Coverage
                # --------------------------------------------------

                defect_pixels = np.count_nonzero(mask_array)

                total_pixels = mask_array.size

                coverage = (defect_pixels / total_pixels) * 100

                self.mask_coverages.append(coverage)

        except Exception:

            self.corrupted_masks.append(mask_path)


    def _verify_pairs(self) -> None:
        """
        Perform final dataset integrity validation.

        This function evaluates whether the dataset is suitable for
        downstream processing. It does NOT judge annotation quality;
        it only checks dataset integrity.
        """

        self.dataset_ready = True

        # --------------------------------------------------
        # Pair Consistency
        # --------------------------------------------------

        if self.total_images != self.total_masks:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Pair Integrity",
                    "message": "Image and mask counts do not match.",
                    "count": abs(self.total_images - self.total_masks),
                }
            )

        if self.total_samples != self.matched_pairs:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Pair Integrity",
                    "message": "Matched pair count is inconsistent.",
                    "count": abs(self.total_samples - self.matched_pairs),
                }
            )

        # --------------------------------------------------
        # Missing Images
        # --------------------------------------------------

        if self.missing_images:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Pair Integrity",
                    "message": "Missing original images.",
                    "count": len(self.missing_images),
                }
            )

        # --------------------------------------------------
        # Missing Masks
        # --------------------------------------------------

        if self.missing_masks:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Pair Integrity",
                    "message": "Missing segmentation masks.",
                    "count": len(self.missing_masks),
                }
            )

        # --------------------------------------------------
        # Resolution Consistency
        # --------------------------------------------------

        if self.resolution_mismatches:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Mask Validation",
                    "message": "Image-mask resolution mismatch.",
                    "count": len(self.resolution_mismatches),
                }
            )

        # --------------------------------------------------
        # Corrupted Images
        # --------------------------------------------------

        if self.corrupted_images:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Image Validation",
                    "message": "Corrupted original images.",
                    "count": len(self.corrupted_images),
                }
            )

        # --------------------------------------------------
        # Corrupted Masks
        # --------------------------------------------------

        if self.corrupted_masks:

            self.dataset_ready = False

            self.audit_issues.append(
                {
                    "severity": "ERROR",
                    "category": "Mask Validation",
                    "message": "Corrupted segmentation masks.",
                    "count": len(self.corrupted_masks),
                }
            )

        # --------------------------------------------------
        # Informational Dataset Statistics
        # --------------------------------------------------

        if self.non_binary_masks:

            self.audit_issues.append(
                {
                    "severity": "INFO",
                    "category": "Mask Statistics",
                    "message": "Non-binary masks detected.",
                    "count": self.non_binary_masks,
                }
            )

        if self.no_defect_masks:

            self.audit_issues.append(
                {
                    "severity": "INFO",
                    "category": "Mask Statistics",
                    "message": "Background-only masks detected.",
                    "count": self.no_defect_masks,
                }
            )

        if self.full_masks:

            self.audit_issues.append(
                {
                    "severity": "INFO",
                    "category": "Mask Statistics",
                    "message": "Completely filled masks detected.",
                    "count": self.full_masks,
                }
            )

    def _print_dataset_summary(self) -> None:
        """
        Print overall dataset summary.
        """

        print("=" * 70)
        print("Dataset Summary")
        print("-" * 70)

        print(f"{'Dataset Root':<30}: {self.dataset_root}")
        print(f"{'Classes':<30}: {self.total_classes}")
        print(f"{'Total Samples':<30}: {self.total_samples}")
        print(f"{'Original Images (JPG)':<30}: {self.total_images}")
        print(f"{'Segmentation Masks (PNG)':<30}: {self.total_masks}")

        if self.total_samples > 0:
            pair_completeness = (
                self.matched_pairs / self.total_samples
            ) * 100
        else:
            pair_completeness = 0.0

        print(f"{'Matched Pairs':<30}: {self.matched_pairs}")
        print(f"{'Pair Completeness':<30}: {pair_completeness:.2f}%")


    def _print_class_distribution(self) -> None:
        """
        Print class distribution statistics.
        """

        print("=" * 70)
        print("Class Distribution")
        print("=" * 70)

        if not self.class_distribution:
            print("No class information available.")
            return

        largest_class = max(
            self.class_distribution,
            key=self.class_distribution.get
        )

        smallest_class = min(
            self.class_distribution,
            key=self.class_distribution.get
        )

        largest_count = self.class_distribution[largest_class]
        smallest_count = self.class_distribution[smallest_class]

        for class_name, count in sorted(self.class_distribution.items()):

            percentage = (count / self.total_samples) * 100

            print(
                f"{class_name:<20}"
                f"{count:>6} "
                f"({percentage:6.2f}%)"
            )

        print("-" * 70)

        print(
            f"{'Largest Class':<25}: "
            f"{largest_class} ({largest_count})"
        )

        print(
            f"{'Smallest Class':<25}: "
            f"{smallest_class} ({smallest_count})"
        )

        imbalance_ratio = (
            largest_count / smallest_count
            if smallest_count > 0
            else float("inf")
        )

        print(
            f"{'Imbalance Ratio':<25}: "
            f"{imbalance_ratio:.2f} : 1"
        )


    def _print_pair_integrity(self) -> None:
        """
        Print image-mask pair integrity report.
        """

        print("=" * 70)
        print("Sample Integrity")
        print("=" * 70)

        print(f"{'Matched Pairs':<30}: {self.matched_pairs}")
        print(f"{'Missing Images':<30}: {len(self.missing_images)}")
        print(f"{'Missing Masks':<30}: {len(self.missing_masks)}")
        print(f"{'Resolution Mismatches':<30}: {len(self.resolution_mismatches)}")
        print(f"{'Corrupted Images':<30}: {len(self.corrupted_images)}")
        print(f"{'Corrupted Masks':<30}: {len(self.corrupted_masks)}")

        print("-" * 70)

        if (
            len(self.missing_images) == 0
            and len(self.missing_masks) == 0
            and len(self.resolution_mismatches) == 0
            and len(self.corrupted_images) == 0
            and len(self.corrupted_masks) == 0
        ):
            print("Status".ljust(30) + ": PASS ✅")

        else:
            print("Status".ljust(30) + ": FAIL ❌")




    def _print_original_image_summary(self) -> None:
        """
        Print summary statistics for the original inspection images.
        """

        print("=" * 70)
        print("Original Image Analysis")
        print("=" * 70)

        # --------------------------------------------------
        # Image Modes
        # --------------------------------------------------

        print("\nImage Modes")
        print("-" * 70)

        for mode, count in sorted(self.image_modes.items()):
            print(f"{mode:<20}{count}")

        # --------------------------------------------------
        # Image Formats
        # --------------------------------------------------

        print("\nImage Formats")
        print("-" * 70)

        for fmt, count in sorted(self.image_formats.items()):
            print(f"{fmt:<20}{count}")

        # --------------------------------------------------
        # Resolution Statistics
        # --------------------------------------------------

        print("\nResolution Statistics")
        print("-" * 70)

        if self.widths:

            print("\nWidth")

            print(f"{'Minimum':<20}: {min(self.widths)}")
            print(f"{'Maximum':<20}: {max(self.widths)}")
            print(f"{'Mean':<20}: {statistics.mean(self.widths):.2f}")
            print(f"{'Median':<20}: {statistics.median(self.widths):.2f}")

            if len(self.widths) > 1:
                print(f"{'Std':<20}: {statistics.stdev(self.widths):.2f}")

        if self.heights:

            print("\nHeight")

            print(f"{'Minimum':<20}: {min(self.heights)}")
            print(f"{'Maximum':<20}: {max(self.heights)}")
            print(f"{'Mean':<20}: {statistics.mean(self.heights):.2f}")
            print(f"{'Median':<20}: {statistics.median(self.heights):.2f}")

            if len(self.heights) > 1:
                print(f"{'Std':<20}: {statistics.stdev(self.heights):.2f}")

        # --------------------------------------------------
        # Aspect Ratio
        # --------------------------------------------------

        if self.aspect_ratios:

            print("\nAspect Ratio")

            print(f"{'Minimum':<20}: {min(self.aspect_ratios):.3f}")
            print(f"{'Maximum':<20}: {max(self.aspect_ratios):.3f}")
            print(f"{'Mean':<20}: {statistics.mean(self.aspect_ratios):.3f}")
            print(f"{'Median':<20}: {statistics.median(self.aspect_ratios):.3f}")

            if len(self.aspect_ratios) > 1:
                print(f"{'Std':<20}: {statistics.stdev(self.aspect_ratios):.3f}")

        # --------------------------------------------------
        # Resolution Frequency
        # --------------------------------------------------

        print("\nUnique Resolutions")
        print("-" * 70)

        print(f"{'Count':<20}: {len(self.image_sizes)}")

        print("\nTop 10 Resolutions")
        print("-" * 70)

        for resolution, count in self.image_sizes.most_common(10):
            print(f"{str(resolution):<20}{count}")


    def _print_mask_summary(self) -> None:
        """
        Print segmentation mask statistics.
        """

        print("=" * 70)
        print("Segmentation Mask Analysis")
        print("=" * 70)

        # --------------------------------------------------
        # Mask Properties
        # --------------------------------------------------

        print("\nMask Modes")
        print("-" * 70)

        for mode, count in sorted(self.mask_modes.items()):
            print(f"{mode:<20}{count}")

        print("\nMask Formats")
        print("-" * 70)

        for fmt, count in sorted(self.mask_formats.items()):
            print(f"{fmt:<20}{count}")

        # --------------------------------------------------
        # Binary Statistics
        # --------------------------------------------------

        print("\nBinary Statistics")
        print("-" * 70)

        print(f"{'Binary Masks':<30}: {self.binary_masks}")
        print(f"{'Non-Binary Masks':<30}: {self.non_binary_masks}")

        binary_percentage = (
            self.binary_masks / self.total_masks * 100
            if self.total_masks else 0
        )

        print(f"{'Binary Percentage':<30}: {binary_percentage:.2f}%")

        # --------------------------------------------------
        # Resolution Consistency
        # --------------------------------------------------

        print("\nResolution Consistency")
        print("-" * 70)

        matching = self.matched_pairs - len(self.resolution_mismatches)

        print(f"{'Matching Resolution':<30}: {matching}")
        print(f"{'Resolution Mismatches':<30}: {len(self.resolution_mismatches)}")

        # --------------------------------------------------
        # Mask Characteristics
        # --------------------------------------------------

        print("\nMask Characteristics")
        print("-" * 70)

        print(f"{'Background-only Masks':<30}: {self.no_defect_masks}")
        print(f"{'Completely Filled Masks':<30}: {self.full_masks}")

        # --------------------------------------------------
        # Coverage Statistics
        # --------------------------------------------------

        if self.mask_coverages:

            print("\nDefect Coverage (%)")
            print("-" * 70)

            print(f"{'Minimum':<20}: {min(self.mask_coverages):.3f}")
            print(f"{'Maximum':<20}: {max(self.mask_coverages):.3f}")
            print(f"{'Mean':<20}: {statistics.mean(self.mask_coverages):.3f}")
            print(f"{'Median':<20}: {statistics.median(self.mask_coverages):.3f}")

            if len(self.mask_coverages) > 1:
                print(f"{'Std':<20}: {statistics.stdev(self.mask_coverages):.3f}")

        # --------------------------------------------------
        # Unique Pixel Value Patterns
        # --------------------------------------------------

        print("\nTop Unique Pixel Value Sets")
        print("-" * 70)

        for values, count in self.unique_mask_value_sets.most_common(10):

            print(f"{str(values):<45}{count}")


    def _print_dataset_readiness(self) -> None:
        """
        Print final dataset readiness report.
        """

        print("=" * 70)
        print("Dataset Readiness")
        print("=" * 70)

        # --------------------------------------------------
        # Overall Status
        # --------------------------------------------------

        status_symbol = "✓" if self.dataset_ready else "✗"
        overall_status = "READY" if self.dataset_ready else "NOT READY"

        print(f"{'Overall Status':<30}: {status_symbol} {overall_status}")

        # --------------------------------------------------
        # Validation Checklist
        # --------------------------------------------------

        print("\nValidation Checklist")
        print("-" * 70)

        checks = [
            (
                "Image-Mask Pair Integrity",
                len(self.missing_images) == 0 and len(self.missing_masks) == 0,
            ),
            (
                "Resolution Consistency",
                len(self.resolution_mismatches) == 0,
            ),
            (
                "Original Image Integrity",
                len(self.corrupted_images) == 0,
            ),
            (
                "Segmentation Mask Integrity",
                len(self.corrupted_masks) == 0,
            ),
        ]

        for check_name, passed in checks:

            symbol = "✓" if passed else "✗"

            print(f"{symbol} {check_name}")

        # --------------------------------------------------
        # Dataset Characteristics
        # --------------------------------------------------

        print("\nDataset Characteristics")
        print("-" * 70)

        print(f"{'Binary Masks':<30}: {self.binary_masks}")
        print(f"{'Non-Binary Masks':<30}: {self.non_binary_masks}")
        print(f"{'Background-only Masks':<30}: {self.no_defect_masks}")
        print(f"{'Completely Filled Masks':<30}: {self.full_masks}")

        # --------------------------------------------------
        # Audit Summary
        # --------------------------------------------------

        print("\nAudit Summary")
        print("-" * 70)

        if not self.audit_issues:

            print("No issues detected.")

        else:

            severity_order = {
                "ERROR": 0,
                "WARNING": 1,
                "INFO": 2,
            }

            issues = sorted(
                self.audit_issues,
                key=lambda x: severity_order.get(x["severity"], 99),
            )

            for issue in issues:

                line = (
                    f"[{issue['severity']}] "
                    f"{issue['category']} - "
                    f"{issue['message']}"
                )

                if "count" in issue:
                    line += f" ({issue['count']})"

                print(line)

        # --------------------------------------------------
        # Recommendation
        # --------------------------------------------------

        print("\nRecommendation")
        print("-" * 70)

        if self.dataset_ready:

            print("✓ Dataset integrity verification completed successfully.")
            print("✓ Dataset is ready for Statistical Exploratory Data Analysis.")
            print("✓ Proceed to Notebook 01 - Statistical EDA.")

        else:

            print("✗ Dataset integrity verification failed.")
            print("Resolve all ERROR level issues before continuing.")



    def _print_report(self) -> None:
        """
        Print the complete dataset audit report.
        """

        print("=" * 70)
        print("VisionInspect AI Dataset Audit")
        print("=" * 70)

        self._print_dataset_summary()
        self._print_class_distribution()
        self._print_pair_integrity()
        self._print_original_image_summary()
        self._print_mask_summary()
        self._print_dataset_readiness()

        print("=" * 70)
        print("Dataset Audit Completed")
        print("=" * 70)


    

