"""GCloud APIs"""

from .compute.instance import create_instance_template
from .compute.instance import instance_template_ctx_mngr
from .compute.instance import create_instance_from_template
from .compute.instance import create_mig_from_template
from .compute.instance import mig_ctx_mngr
