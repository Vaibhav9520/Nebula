document.addEventListener("DOMContentLoaded", function() {
    // Sample data for energy consumption trends
    const labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    const energyData = [120, 150, 180, 210, 250, 300, 270, 260, 240, 230, 200, 180];

    // Generate the chart
    const ctx = document.getElementById("energyChart").getContext("2d");
    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Energy Consumption (kWh)",
                data: energyData,
                borderColor: "#243B55",
                backgroundColor: "rgba(36, 59, 85, 0.2)",
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: "#243B55"
                    }
                }
            }
        }
    });

    // Table filter function
    document.getElementById("filterInput").addEventListener("keyup", function() {
        let filter = this.value.toLowerCase();
        let rows = document.querySelectorAll("#dataTable tbody tr");

        rows.forEach(row => {
            let text = row.innerText.toLowerCase();
            row.style.display = text.includes(filter) ? "" : "none";
        });
    });
});
