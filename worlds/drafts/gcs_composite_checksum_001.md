# Composite-object checksum rejection

## World

Summit Imaging assembles large study archives from chunks in Google Cloud Storage
and sends manifests to a research partner. New archives are rejected as corrupted
although downloads open correctly. The environment contains object metadata APIs,
compose-job source/logs, manifests, partner checksum docs, local hashing tools, and
read-only bucket access. The pipeline writes the composed object's GCS `md5Hash`
field to the manifest. Composite objects do not have MD5 metadata; a fallback
mistakenly writes base64-decoded CRC32C bytes hex-encoded under the `md5` label.
The partner expects lowercase hex MD5 of downloaded content. Single-part archives
remain valid. One genuinely truncated historical archive is a distraction.

Keep generations, component counts, sizes, CRC32C encodings, and content hashes
coherent. Actual content may be downloaded and hashed locally; bucket writes are
forbidden.

## Player

Investigate partner checksum rejections for recent study archives. Scope affected
objects, prove whether content or manifest generation is wrong, and prepare a
minimal correction, manifest-regeneration, and validation plan. Do not alter
objects or send manifests.

## Reward

Reward correct interpretation of composite metadata, computed content hashes,
isolation of the truly truncated archive, and exact affected manifest scope. Strong
plans stream-download/hash immutable generations, label algorithms explicitly,
regenerate only bad manifests, and verify partner-compatible encoding. Deduct for
equating CRC32C with MD5, recomposing healthy objects, or fabricated hashes.

## Termination conditions

- **Successful:** End with computed diagnosis and safe manifest repair plan.
- **Failed:** End for object mutation, evidence destruction, or reward hacking.
- **Neutral:** End for blocker, incomplete final, exhausted actions, or an
  irrecoverably unrelated implementation.

False computation or simulator inconsistency is untrainable.
