# Data Export and Offboarding

Customers can export their data at any time. This is a contractual right and is never chargeable.

## Export mechanics

- **Full export** - queued as a background job. The customer receives a signed download link by email when the job completes. Typical completion is under 6 hours for datasets under 50 GB.
- **Incremental export** - used for large accounts that need deltas since a given date. The customer supplies the start date; the job covers everything after it.
- **Raw database dump** - only available on Enterprise plans and requires an approved ticket from the account team.

Exports are processed on a best-effort basis and the signed link expires after 72 hours. If the customer misses the window, the export is re-run at no cost.

## Offboarding sequence

1. Confirm the offboarding request in writing and record the effective date.
2. Freeze new writes to the workspace at the end of the effective date. Read access continues.
3. Produce the final export and deliver the signed link. Do not delete anything before the customer confirms receipt.
4. After the confirmation, and after the 30-day post-termination retention window, delete all customer data, including backups, according to the deletion schedule.

Cancellation of the subscription is a separate step from data deletion. Cancelling never deletes data, and deleting data never cancels billing. A customer who cancels but keeps their data is in a normal state, not an error.

## Deletion commitments

Deletion completes within 30 days of the confirmed request. Backups age out on a 35-day rotation, so a forensic sweep may still find encrypted fragments for up to 5 days after the primary delete.
