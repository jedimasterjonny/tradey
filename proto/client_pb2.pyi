from collections.abc import Sequence

from google.protobuf.message import Message

class PHistoricalPrice(Message):
    date: int
    close: int

class PSecurity(Message):
    uuid: str
    name: str
    currencyCode: str
    prices: Sequence[PHistoricalPrice]

class _PAssignment(Message):
    investmentVehicle: str
    weight: int
    rank: int

class _PClassification(Message):
    id: str
    parentId: str
    name: str
    weight: int
    rank: int
    assignments: Sequence[_PAssignment]

class PTaxonomy(Message):
    id: str
    name: str
    classifications: Sequence[_PClassification]

class PTransaction(Message):
    uuid: str
    type: int
    security: str
    shares: int

class PClient(Message):
    version: int
    baseCurrency: str
    securities: Sequence[PSecurity]
    transactions: Sequence[PTransaction]
    taxonomies: Sequence[PTaxonomy]
