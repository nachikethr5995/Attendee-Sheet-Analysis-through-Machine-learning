"""Signature Verification Service.

This service processes detected signatures from layout detection and verifies them
using GPDS verification model. It acts as a post-detection verification layer.
"""

# Import core first to ensure path fix is applied
import core  # This ensures user site-packages is in path

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from core.logging import log
from core.config import settings
from core.utils import (
    load_image_from_canonical_id,
    get_intermediate_json_path
)
from postprocessing.signature_verifier import GPDSSignatureVerifier


class SignatureVerificationService:
    """Service for verifying detected signatures.
    
    This service:
    1. Takes detected signature regions from layout detection
    2. Crops signature regions from the image
    3. Verifies each crop using GPDS verification model
    4. Returns verification results with original bounding boxes
    """
    
    def __init__(self, verification_threshold: float = 0.7):
        """Initialize signature verification service.
        
        Args:
            verification_threshold: Confidence threshold for valid signature
        """
        self.verification_threshold = verification_threshold
        self.verifier = GPDSSignatureVerifier(
            verification_threshold=verification_threshold,
            use_gpu=settings.USE_GPU
        )
        
        log.info("Signature Verification Service initialized")
        log.info(f"  GPDS verifier available: {self.verifier.is_available()}")
        log.info(f"  Verification threshold: {verification_threshold}")
    
    def verify_signatures(self,
                         signatures: List[Dict[str, Any]],
                         file_id: Optional[str] = None,
                         pre_0_id: Optional[str] = None,
                         pre_01_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Verify detected signatures.
        
        Args:
            signatures: List of signature detections from layout service
                       Each dict should have 'bbox' (normalized coordinates)
            file_id: Raw file identifier
            pre_0_id: Basic preprocessing identifier
            pre_01_id: Advanced preprocessing identifier
            
        Returns:
            list: Verified signatures with verification results
        """
        if not signatures:
            log.info("No signatures to verify")
            return []
        
        if not self.verifier.is_available():
            log.warning("GPDS verifier not available, skipping verification")
            # Return signatures with verification skipped
            verified = []
            for i, sig in enumerate(signatures):
                verified.append({
                    **sig,
                    'signature_id': f"sig_{i+1:02d}",
                    'verification_score': 0.0,
                    'is_valid_signature': False,
                    'verification_model': 'GPDS',
                    'verification_status': 'skipped',
                    'verification_notes': 'GPDS model not available'
                })
            return verified
        
        # Load image
        try:
            image = load_image_from_canonical_id(
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            width, height = image.size
            log.info(f"Loaded image for verification: {width}x{height}")
        except Exception as e:
            log.error(f"Failed to load image for verification: {str(e)}")
            # Return signatures with verification error
            verified = []
            for i, sig in enumerate(signatures):
                verified.append({
                    **sig,
                    'signature_id': f"sig_{i+1:02d}",
                    'verification_score': 0.0,
                    'is_valid_signature': False,
                    'verification_model': 'GPDS',
                    'verification_status': 'error',
                    'verification_notes': f'Image load failed: {str(e)}'
                })
            return verified
        
        # Process each signature
        verified_signatures = []
        
        for i, signature in enumerate(signatures):
            signature_id = f"sig_{i+1:02d}"
            bbox = signature.get('bbox', [])
            
            if not bbox or len(bbox) < 4:
                log.warning(f"Signature {signature_id} has invalid bbox, skipping")
                verified_signatures.append({
                    **signature,
                    'signature_id': signature_id,
                    'verification_score': 0.0,
                    'is_valid_signature': False,
                    'verification_model': 'GPDS',
                    'verification_status': 'error',
                    'verification_notes': 'Invalid bounding box'
                })
                continue
            
            # Convert normalized bbox to pixel coordinates
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)
            
            # Crop signature region
            try:
                signature_crop = image.crop((x1, y1, x2, y2))
                log.debug(f"Verifying signature {signature_id}: crop size {signature_crop.size}")
            except Exception as e:
                log.warning(f"Failed to crop signature {signature_id}: {str(e)}")
                verified_signatures.append({
                    **signature,
                    'signature_id': signature_id,
                    'verification_score': 0.0,
                    'is_valid_signature': False,
                    'verification_model': 'GPDS',
                    'verification_status': 'error',
                    'verification_notes': f'Crop failed: {str(e)}'
                })
                continue
            
            # Verify signature
            verification_result = self.verifier.verify(signature_crop)
            
            # Merge verification results with original signature data
            verified_sig = {
                **signature,  # Preserve original detection data
                'signature_id': signature_id,
                'verification_score': verification_result['verification_score'],
                'is_valid_signature': verification_result['is_valid_signature'],
                'verification_model': verification_result['model'],
                'verification_status': verification_result['status'],
                'verification_notes': verification_result['notes']
            }
            
            verified_signatures.append(verified_sig)
            
            log.debug(f"Signature {signature_id}: score={verification_result['verification_score']:.2f}, "
                     f"valid={verification_result['is_valid_signature']}")
        
        # Log summary
        valid_count = sum(1 for sig in verified_signatures if sig.get('is_valid_signature', False))
        log.info(f"Signature verification complete: {valid_count}/{len(verified_signatures)} valid signatures")
        
        return verified_signatures
    
    def verify_from_layout(self,
                          canonical_id: str,
                          file_id: Optional[str] = None,
                          pre_0_id: Optional[str] = None,
                          pre_01_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Verify signatures from layout detection JSON.
        
        Args:
            canonical_id: Canonical ID (pre_0_id or pre_01_id)
            file_id: Raw file identifier
            pre_0_id: Basic preprocessing identifier
            pre_01_id: Advanced preprocessing identifier
            
        Returns:
            list: Verified signatures
        """
        # Load layout JSON
        layout_path = get_intermediate_json_path(canonical_id, 'layout')
        
        if not layout_path.exists():
            log.warning(f"Layout JSON not found: {layout_path}")
            return []
        
        try:
            with open(layout_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            
            signatures = layout_data.get('signatures', [])
            
            if not signatures:
                log.info("No signatures found in layout JSON")
                return []
            
            log.info(f"Found {len(signatures)} signatures in layout JSON")
            
            # Verify signatures
            return self.verify_signatures(
                signatures=signatures,
                file_id=file_id,
                pre_0_id=pre_0_id,
                pre_01_id=pre_01_id
            )
            
        except Exception as e:
            log.error(f"Failed to load layout JSON: {str(e)}", exc_info=True)
            return []




