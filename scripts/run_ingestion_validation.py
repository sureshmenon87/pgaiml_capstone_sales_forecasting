from src.datascience.ingestion import (
    load_restaurants,
    load_items,
    load_sales,
)
from src.datascience.validation import (
    validate_restaurants,
    validate_items,
    validate_sales,
)


def main():
    restaurants = load_restaurants()
    items = load_items()
    sales = load_sales()

    validate_restaurants(restaurants)
    validate_items(items, restaurants)
    validate_sales(sales)

    print("Ingestion and validation completed successfully")


if __name__ == "__main__":
    main()
