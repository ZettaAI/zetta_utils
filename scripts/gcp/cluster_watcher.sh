 #!/bin/bash
TERM_COUNT=$(kubectl get pods | grep Term | wc -l)
RUN_COUNT=$(kubectl get pods | grep Run | wc -l)
PEND_COUNT=$(kubectl get pods | grep Pend | wc -l)
ERR_COUNT=$(kubectl get pods | grep Err | wc -l)
CREA_COUNT=$(kubectl get pods | grep Cont| wc -l)

echo Running: ${RUN_COUNT}
echo Pending: ${PEND_COUNT}
echo Error: ${ERR_COUNT}
echo Creating: ${CREA_COUNT}
echo Termindating: ${TERM_COUNT}
