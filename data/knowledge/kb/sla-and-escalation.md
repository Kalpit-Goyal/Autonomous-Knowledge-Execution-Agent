# SLA and Escalation

Every ticket carries a priority. Priority is not a customer courtesy label - it drives response commitments, escalation targets, and who gets notified.

## Priority definitions

- **P1 Critical** - the service is down or unusable for the customer. First response 15 minutes, resolution target 4 hours. Notifies the platform on-call, the engineering manager, and the account executive. Pages the on-call rotation.
- **P2 Major** - a core feature is degraded and there is no workaround. First response 60 minutes, resolution target 8 hours. Notifies platform on-call and the account executive. Pages the on-call rotation.
- **P3 Minor** - partial degradation with a workaround available. First response 4 hours, resolution target 24 hours. Notifies the support queue only. No page.
- **P4 Low** - cosmetic issues, general questions, and feature requests. First response 8 hours, resolution target 72 hours. No notification.

## Business hours and clock stops

SLA clocks run during business hours only: 09:00 to 18:00 UTC, Monday to Friday.

A clock stops while a ticket is in one of these states:

- `awaiting_customer_response`
- `awaiting_third_party_provider`

A ticket sitting in `awaiting_customer_response` must not be reported as breached. When the customer replies, the clock resumes and the accumulated stop time is subtracted.

## Automatic escalation

When a ticket's open time exceeds the resolution target for its priority, it escalates automatically. Tier P1 and P2 tickets page the on-call rotation at the escalation threshold, not only at the resolution target.

## Security incidents

Any suspected data breach, credential leak, or unauthorised access escalates to the security-incident channel at P1 regardless of how many customers are affected. Do not wait for customer impact to be demonstrated.
