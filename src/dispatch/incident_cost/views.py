from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from dispatch.database.core import get_db
from dispatch.database.service import common_parameters, search_filter_sort_paginate
from dispatch.auth.permissions import SensitiveProjectActionPermission, PermissionsDependency
from dispatch.models import PrimaryKey

from .models import (
    IncidentCostCreate,
    IncidentCostPagination,
    IncidentCostRead,
    IncidentCostUpdate,
)
from .service import create, delete, get, update


router = APIRouter()


@router.get("", response_model=IncidentCostPagination)
def get_incident_costs(*, common: dict = Depends(common_parameters)):
    """Get all incident costs, or only those matching a given search term."""
    return search_filter_sort_paginate(model="IncidentCost", **common)


@router.get("/{incident_cost_id}", response_model=IncidentCostRead)
def get_incident_cost(*, db_session: Session = Depends(get_db), incident_cost_id: PrimaryKey):
    """Get an incident cost by its id."""
    if incident_cost := get(
        db_session=db_session, incident_cost_id=incident_cost_id
    ):
        return incident_cost
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=[{"msg": "An incident cost with this id does not exist."}],
        )


@router.post(
    "",
    response_model=IncidentCostRead,
    dependencies=[Depends(PermissionsDependency([SensitiveProjectActionPermission]))],
)
def create_incident_cost(
    *, db_session: Session = Depends(get_db), incident_cost_in: IncidentCostCreate
):
    """Create an incident cost."""
    return create(db_session=db_session, incident_cost_in=incident_cost_in)


@router.put(
    "/{incident_cost_id}",
    response_model=IncidentCostRead,
    dependencies=[Depends(PermissionsDependency([SensitiveProjectActionPermission]))],
)
def update_incident_cost(
    *,
    db_session: Session = Depends(get_db),
    incident_cost_id: PrimaryKey,
    incident_cost_in: IncidentCostUpdate,
):
    """Update an incident cost by its id."""
    incident_cost = get(db_session=db_session, incident_cost_id=incident_cost_id)
    if not incident_cost:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=[{"msg": "An incident cost with this id does not exist."}],
        )
    incident_cost = update(
        db_session=db_session,
        incident_cost=incident_cost,
        incident_cost_in=incident_cost_in,
    )
    return incident_cost


@router.delete(
    "/{incident_cost_id}",
    dependencies=[Depends(PermissionsDependency([SensitiveProjectActionPermission]))],
)
def delete_incident_cost(*, db_session: Session = Depends(get_db), incident_cost_id: PrimaryKey):
    """Delete an incident cost, returning only an HTTP 200 OK if successful."""
    if incident_cost := get(
        db_session=db_session, incident_cost_id=incident_cost_id
    ):
        delete(db_session=db_session, incident_cost_id=incident_cost_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=[{"msg": "An incident cost with this id does not exist."}],
        )
