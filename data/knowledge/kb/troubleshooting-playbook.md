# Troubleshooting Playbook

This playbook covers the failure modes support sees most. Work top down and stop when the symptom resolves.

## Login loops after password reset

**Symptom:** customer enters correct credentials and is bounced back to the login form.

1. Confirm the workspace domain. A user on a personal address that was never verified will loop silently.
2. Check for an active SSO enforcement flag on the workspace. If SSO is enforced, password auth is rejected by design and the loop is expected.
3. Check `session_store` for a stale session older than 24 hours and clear it.
4. If SSO is enforced and the user has not been provisioned in the identity provider, the correct fix is provisioning, not a password reset. Escalate to the account team.

## Webhooks not delivering

**Symptom:** events are created in the product but the customer endpoint never receives them.

1. Confirm the endpoint is reachable from the public internet. Internal and localhost endpoints are never called.
2. Check the delivery log for `410 Gone`. A `410` is permanent - the customer must re-register the endpoint, retrying will never succeed.
3. Check for `429` responses. A `429` is a backoff signal, not a failure; the queue retries with exponential backoff.
4. Check TLS expiry on the customer endpoint. An expired certificate produces a TLS handshake error, not an HTTP status.

## Slow dashboards

**Symptom:** dashboards load, but slowly, usually after the workspace grows past a few thousand records.

1. Check the plan against current usage. The usual cause is an account on a plan tier below its data volume.
2. Check for unindexed custom fields. Each unindexed field multiplies query time on list views.
3. Check for a stuck export job. A long-running export competes for the same query budget.

## Duplicate records after a sync

**Symptom:** customer reports duplicated rows following a connector sync.

1. Identify the connector and the last successful run.
2. Check the cursor. A reset cursor replays history and duplicates rows; the correct repair is to reset the cursor to the last committed position, not to delete the duplicates by hand.
3. If the connector is third-party, move the ticket to `awaiting_third_party_provider` so the SLA clock stops while they investigate.

## When to stop troubleshooting

Stop and escalate rather than continuing to try fixes when any of these are true:

- The symptom implicates data loss or corruption.
- Two different fixes appear to have changed the behaviour, so the root cause is no longer isolated.
- The reproduction is intermittent and cannot be triggered on demand.
- The customer is a security-relevant account and the issue touches authentication or access control.
