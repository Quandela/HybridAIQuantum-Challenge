import os

import numpy as np

# /!\ Ensure version 0.12.0 is installed
import perceval as pcvl
import perceval.providers.scaleway as scw

# Scaleway: scaleway.com
# QaaS product page: labs.scaleway.com/en/qaas
# QaaS integration Perceval documentation and examples: perceval.quandela.net/docs/v0.12/providers.html
# QaaS API: scaleway.com/en/developers/api/qaas

# # UUID / Can be found at console.scaleway.com/project/settings
# __SCW_PROJECT_ID = os.environ["c618fb5e-7a27-4c3f-8ef7-032dfed8977d"]

# # UUID / Can be generated at console.scaleway.com/iam/api-keys
# __SCW_TOKEN = os.environ["52342b1c-4fb2-47fa-820a-0165067c7b74"]

# A session is a time limited access to a QPU (either real or emulated)
# You can see your running sessions at console.scaleway.com/qaas/sessions
session = scw.Session(
    project_id="c618fb5e-7a27-4c3f-8ef7-032dfed8977d",
    token="52342b1c-4fb2-47fa-820a-0165067c7b74",
    platform="qpu:ascella",  # Simulation on Nvidia L4 GPU
    max_idle_duration_s=1200,  # session is automatically revoked after 6 minutes without any new job
    max_duration_s=3600  # session is automatically revoked after one hour
)

## The algorithm part
cnot_circuit = pcvl.Circuit(6, name="Ralph CNOT")
cnot_circuit.add(
    (0, 1),
    pcvl.BS.H(
        pcvl.BS.r_to_theta(1 / 3),
        phi_tl=-np.pi / 2,
        phi_bl=np.pi,
        phi_tr=np.pi / 2,
    ),
)
cnot_circuit.add((3, 4), pcvl.BS.H())
cnot_circuit.add(
    (2, 3),
    pcvl.BS.H(
        pcvl.BS.r_to_theta(1 / 3),
        phi_tl=-np.pi / 2,
        phi_bl=np.pi,
        phi_tr=np.pi / 2,
    ),
)
cnot_circuit.add((4, 5), pcvl.BS.H(pcvl.BS.r_to_theta(1 / 3)))
cnot_circuit.add((3, 4), pcvl.BS.H())
## End of the algorithm part

try:
    # Start a new QPU session (or do nothing if a *running* session with same deduplication_id already exists)
    session.start()

    processor = session.build_remote_processor()

    processor.set_circuit(cnot_circuit)
    processor.min_detected_photons_filter(3)
    processor.with_input(pcvl.BasicState("|0,1,0,1,0,1>"))

    # Target platform (sim:sampling:l4) will do sampling
    # it requires to run many times like real photonic QPUs
    n_samples = 1000
    sampler = pcvl.algorithm.Sampler(processor, max_shots_per_call=n_samples)

    # Sync call to run the job
    job = sampler.probs(n_samples)

    # Get the total probabilities of possible distributions
    output_distribution = job.get("results")

    print(output_distribution)
finally:
    # Order to end the running session
    # If not done, session will be revoked after max_duration or max_idle_duration is reached
    #
    # /!\ Beware, a running session is billed per minute (according to the underlying GPU)
    # ==> Keep in mind to revoke any session if you don't use it to not waste your credits!
    # You can revoked them manually at console.scaleway.com/qaas/sessions
    # At any time you can follow your consumption at console.scaleway.com/billing/consumption
    session.stop()
